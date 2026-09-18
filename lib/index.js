/**
 * dsh-plugin-c2c — Cache-to-Cache (C2C) for DeepSeek Harness.
 *
 * A host-plane plugin that registers the `c2c_*` tools plus one usage section
 * in the global system prompt. The tools speak HTTP to a local C2C daemon
 * (`python/c2c_daemon.py`) that owns the `rosetta` runtime — the PyTorch code
 * from https://github.com/thu-nics/C2C that fuses one model's KV-cache into
 * another's instead of exchanging text.
 *
 * Split of responsibilities: this module never imports Python or torch. It
 * starts the daemon on demand, waits for readiness, and forwards calls. Model
 * loading therefore survives tool timeouts and repeated calls reuse warm
 * weights instead of paying the load cost again.
 *
 * Installation (bundle): `dsh plugin --profile <name> add <this package>`.
 *
 * @module dsh-plugin-c2c
 */

import z from '@deepseek-ai/schemastery';
import { defineTool } from '@deepseek-ai/dsh-tools';
import { spawn } from 'node:child_process';
import { existsSync, openSync } from 'node:fs';
import { mkdir, readFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const name = 'c2c';

/** Services this plugin consumes. `tools` and `systemPrompt` are required. */
export const inject = ['tools', 'systemPrompt'];

export const Config = z.object({
	pythonPath: z.string().default(''),
	repoRoot: z.string().default(''),
	port: z.natural().default(8765),
	host: z.string().default('127.0.0.1'),
	idleUnloadSeconds: z.natural().default(900),
	loadTimeoutSeconds: z.natural().default(600),
	requestTimeoutSeconds: z.natural().default(900),
	logPath: z.string().default(''),
	promptSectionOrder: z.natural().default(118),
});

/** Tool names this plugin registers, in a stable order. */
export const TOOL_NAMES = ['c2c_status', 'c2c_models', 'c2c_load', 'c2c_chat', 'c2c_compare', 'c2c_unload'];

const here = dirname(fileURLToPath(import.meta.url));

/** Absolute path to the bundled daemon script. */
function daemonScript() {
	// lib/index.js -> ../python/c2c_daemon.py
	return resolve(here, '..', 'python', 'c2c_daemon.py');
}

/**
 * Candidate interpreters, best first. The venv created by the reproduction
 * step is preferred because a bare `python3` rarely has torch installed.
 * @param {string} explicit - configured pythonPath, may be empty.
 * @param {string} repoRoot - reproduced C2C checkout, may be empty.
 * @returns {string[]} absolute interpreter paths to probe in order.
 */
function interpreterCandidates(explicit, repoRoot) {
	const list = [];
	if (explicit) list.push(isAbsolute(explicit) ? explicit : resolve(process.cwd(), explicit));
	if (repoRoot) {
		const root = isAbsolute(repoRoot) ? repoRoot : resolve(process.cwd(), repoRoot);
		list.push(join(root, '.venv', 'bin', 'python'));
		list.push(join(root, '.venv', 'bin', 'python3'));
		list.push(join(root, 'venv', 'bin', 'python'));
	}
	list.push('/opt/homebrew/bin/python3', '/usr/local/bin/python3', '/usr/bin/python3', 'python3');
	return list;
}

/**
 * Resolve the first interpreter that exists on disk.
 * @param {string} explicit - configured pythonPath.
 * @param {string} repoRoot - reproduced C2C checkout.
 * @returns {string} an interpreter path or command name.
 */
function resolveInterpreter(explicit, repoRoot) {
	for (const candidate of interpreterCandidates(explicit, repoRoot)) {
		if (!candidate.includes('/')) return candidate; // bare command: let PATH decide
		if (existsSync(candidate)) return candidate;
	}
	return 'python3';
}

/**
 * Resolve the configured repo root, falling back to common locations.
 * @param {string} configured - configured repoRoot.
 * @returns {string} an absolute path, possibly non-existent.
 */
function resolveRepoRoot(configured) {
	if (configured) return isAbsolute(configured) ? configured : resolve(process.cwd(), configured);
	const cwd = process.cwd();
	const candidates = [
		// Sibling of this checkout, e.g. <ws>/c2c next to <ws>/dsh-plugin-c2c.
		resolve(here, '..', '..', 'c2c'),
		// DSH runs with the session workspace as cwd, so a `c2c` directory there
		// is the most likely layout for someone who reproduced it locally.
		join(cwd, 'c2c'),
		join(homedir(), 'factor_digging', 'c2c'),
		join(homedir(), 'c2c'),
	];
	for (const candidate of candidates) {
		if (existsSync(join(candidate, 'rosetta'))) return candidate;
	}
	return candidates[0];
}

/**
 * Runtime handle shared by every tool body: owns the daemon process, its
 * readiness state, and the HTTP client used to talk to it.
 */
class C2CClient {
	/**
	 * @param {object} config - validated plugin config.
	 * @param {object} ctx - cordis context, used for logging.
	 */
	constructor(config, ctx) {
		this.config = config;
		this.ctx = ctx;
		this.repoRoot = resolveRepoRoot(config.repoRoot);
		this.python = resolveInterpreter(config.pythonPath, this.repoRoot);
		this.script = daemonScript();
		this.base = `http://${config.host}:${config.port}`;
		this.child = null;
		this.starting = null;
		this.logPath = config.logPath
			? (isAbsolute(config.logPath) ? config.logPath : resolve(process.cwd(), config.logPath))
			: join(tmpdir(), `dsh-c2c-daemon-${config.port}.log`);
	}

	/** Human-readable description of the resolved runtime, for diagnostics. */
	describe() {
		return {
			python: this.python,
			script: this.script,
			repoRoot: this.repoRoot,
			baseUrl: this.base,
			logPath: this.logPath,
			daemonRunning: this.child !== null && this.child.exitCode === null,
		};
	}

	/**
	 * One HTTP round trip with an abort-based timeout.
	 * @param {string} path - route, e.g. '/status'.
	 * @param {object} [options] - method and body.
	 * @returns {Promise<object>} parsed JSON payload.
	 */
	async request(path, options = {}) {
		const { method = 'GET', body, timeoutMs = this.config.requestTimeoutSeconds * 1000 } = options;
		const controller = new AbortController();
		const timer = setTimeout(() => controller.abort(), timeoutMs);
		try {
			const response = await fetch(`${this.base}${path}`, {
				method,
				headers: body === undefined ? undefined : { 'content-type': 'application/json' },
				body: body === undefined ? undefined : JSON.stringify(body),
				signal: controller.signal,
			});
			const text = await response.text();
			let payload;
			try {
				payload = text ? JSON.parse(text) : {};
			} catch {
				payload = { ok: false, error: `non-JSON response (HTTP ${response.status}): ${text.slice(0, 400)}` };
			}
			if (!response.ok || payload.ok === false) {
				throw new Error(payload.error ?? `C2C daemon returned HTTP ${response.status}`);
			}
			return payload;
		} catch (error) {
			if (error.name === 'AbortError') throw new Error(`C2C daemon request timed out after ${timeoutMs}ms (${path})`);
			throw error;
		} finally {
			clearTimeout(timer);
		}
	}

	/**
	 * Probe the daemon's health endpoint.
	 * @returns {Promise<object|null>} status payload, or null when unreachable.
	 */
	async probe() {
		try {
			return await this.request('/status', { timeoutMs: 3000 });
		} catch {
			return null;
		}
	}

	/**
	 * Ensure the daemon is running and healthy, spawning it when necessary.
	 * Concurrent callers share one startup attempt.
	 * @returns {Promise<object>} the daemon status payload.
	 */
	async ensureRunning() {
		const alive = await this.probe();
		if (alive) return alive;
		if (this.starting) return this.starting;
		this.starting = this.spawnDaemon().finally(() => {
			this.starting = null;
		});
		return this.starting;
	}

	/**
	 * Spawn the daemon detached from the agent's lifetime and wait for readiness.
	 * @returns {Promise<object>} the first healthy status payload.
	 */
	async spawnDaemon() {
		if (!existsSync(this.script)) throw new Error(`C2C daemon script not found at ${this.script}`);
		await mkdir(dirname(this.logPath), { recursive: true });
		const out = openSync(this.logPath, 'a');
		const args = [this.script, '--host', this.config.host, '--port', String(this.config.port), '--idle-unload', String(this.config.idleUnloadSeconds)];
		if (this.repoRoot && existsSync(join(this.repoRoot, 'rosetta'))) args.push('--repo-root', this.repoRoot);
		this.ctx?.logger?.info?.(`c2c: starting daemon ${this.python} ${args.join(' ')}`);
		const child = spawn(this.python, args, { detached: true, stdio: ['ignore', out, out] });
		child.unref();
		this.child = child;
		const deadline = Date.now() + this.config.loadTimeoutSeconds * 1000;
		let lastError = 'daemon did not become healthy';
		while (Date.now() < deadline) {
			await new Promise((r) => setTimeout(r, 500));
			const status = await this.probe();
			if (status) return status;
			if (child.exitCode !== null) {
				lastError = `daemon exited with code ${child.exitCode}; see ${this.logPath}`;
				break;
			}
		}
		const tail = await this.readLogTail();
		throw new Error(`${lastError}\n--- daemon log tail (${this.logPath}) ---\n${tail}`);
	}

	/**
	 * Read the last lines of the daemon log to explain a failed startup.
	 * @returns {Promise<string>} log tail, or a placeholder when unreadable.
	 */
	async readLogTail() {
		try {
			const content = await readFile(this.logPath, 'utf8');
			return content.split('\n').slice(-40).join('\n');
		} catch {
			return '(no log available)';
		}
	}

	/**
	 * Load a model pair, tolerating a cold load that outlives the HTTP timeout
	 * by falling back to a longer retry once the daemon reports it is loaded.
	 * @param {object} body - /load request body.
	 * @returns {Promise<object>} load result.
	 */
	async load(body) {
		await this.ensureRunning();
		const timeoutMs = this.config.loadTimeoutSeconds * 1000;
		return this.request('/load', { method: 'POST', body, timeoutMs });
	}

	/**
	 * Run one generation.
	 * @param {object} body - /generate request body.
	 * @returns {Promise<object>} generation result.
	 */
	async generate(body) {
		await this.ensureRunning();
		return this.request('/generate', { method: 'POST', body });
	}
}

/**
 * Render a generation result as model-facing text.
 * @param {object} value - canonical generation value.
 * @returns {string} the rendered block.
 */
function renderGeneration(value) {
	const lines = [
		`pair: ${value.pair}    mode: ${value.mode}    device: ${value.device}`,
		`tokens: ${value.generated_tokens} generated from ${value.prompt_tokens} prompt tokens in ${value.seconds}s`,
	];
	if (value.tokens_per_second) lines.push(`speed: ${value.tokens_per_second} tok/s`);
	if (value.baseline) {
		lines.push('', '--- receiver only (no C2C) ---', value.baseline.text);
		lines.push('', '--- C2C fused ---');
	}
	lines.push(value.text);
	return lines.join('\n');
}

/**
 * Register every `c2c_*` tool into the shared registry.
 * @param {object} ctx - cordis context.
 * @param {object} config - validated plugin config.
 */
function registerTools(ctx, config) {
	const client = new C2CClient(config, ctx);

	ctx.tools.register(defineTool({
		name: 'c2c_status',
		description:
			'Report the local Cache-to-Cache (C2C) runtime: whether a model pair is loaded, which receiver/sharer models and projector checkpoints are in use, the resolved device and dtype, and the interpreter/daemon paths. Use this first when a C2C call fails, to see whether the daemon is reachable.',
		parameters: {
			start: {
				type: 'boolean',
				description: 'Start the daemon if it is not running. Defaults to false, which only probes.',
			},
		},
		output: {
			schema: { type: 'json' },
			render: (_args, value) => [{ type: 'text', text: JSON.stringify(value, null, 2) }],
		},
		async execute(args) {
			const runtime = client.describe();
			if (!args.start) {
				const status = await client.probe();
				return { ...runtime, reachable: status !== null, status: status ?? null };
			}
			const status = await client.ensureRunning();
			return { ...runtime, reachable: true, status };
		},
	}));

	ctx.tools.register(defineTool({
		name: 'c2c_models',
		description:
			'List the C2C fuser model pairs published on Hugging Face (nics-efc/C2C_Fuser) with their receiver and sharer models, and report which pair is currently loaded. Sizes range from 0.6B+0.5B up to 8B+7B.',
		parameters: {},
		output: {
			schema: { type: 'json' },
			render: (_args, value) => [{ type: 'text', text: JSON.stringify(value, null, 2) }],
		},
		async execute() {
			await client.ensureRunning();
			const catalog = await client.request('/models');
			const status = await client.request('/status');
			return { ...catalog, loaded: status.loaded ? status : null };
		},
	}));

	ctx.tools.register(defineTool({
		name: 'c2c_load',
		description:
			'Load a C2C model pair into the local daemon. Downloads the receiver (base) and sharer (teacher) weights from Hugging Face plus the trained projector checkpoints on first use, then builds the rosetta fusion model. Loading a pair takes tens of seconds on a laptop; the daemon keeps it warm afterwards and releases it after the configured idle period.',
		parameters: {
			pair: {
				type: 'string',
				description: 'Registry pair name, e.g. "qwen3_0.6b+qwen2.5_0.5b". Defaults to the smallest published pair.',
			},
			base_model: {
				type: 'string',
				description: 'Override the receiver model id or local path (advanced; pair must then be omitted or custom).',
			},
			teacher_model: {
				type: 'string',
				description: 'Override the sharer model id or local path (advanced).',
			},
			checkpoints_dir: {
				type: 'string',
				description: 'Directory holding projector_*.json/.pt and projector_config.json. Defaults to the registry pair checkpoint.',
			},
			device: {
				type: 'string',
				description: 'auto, mps, cuda, or cpu. Defaults to auto (cuda, then mps, then cpu).',
			},
			dtype: {
				type: 'string',
				description: 'auto, float16, bfloat16, or float32. Defaults to auto (bfloat16 on cuda, float16 on mps, float32 on cpu).',
			},
		},
		output: {
			schema: { type: 'json' },
			render: (_args, value) => [{ type: 'text', text: JSON.stringify(value, null, 2) }],
		},
		async execute(args) {
			const body = {};
			for (const key of ['pair', 'base_model', 'teacher_model', 'checkpoints_dir', 'device', 'dtype']) {
				if (args[key] !== undefined && args[key] !== '') body[key] = args[key];
			}
			const result = await client.load(body);
			return result.status ?? result;
		},
	}));

	ctx.tools.register(defineTool({
		name: 'c2c_chat',
		description:
			'Generate an answer with the loaded C2C model pair. In mode "c2c" (default) the sharer model\'s KV-cache is projected and fused into the receiver during the prompt pass, so the receiver answers with combined latent knowledge instead of a relayed text summary. In mode "baseline" the sharer is bypassed entirely and the receiver answers alone, which is the control for comparison. Requires a loaded pair; call c2c_load first.',
		parameters: {
			prompt: {
				type: 'string',
				required: true,
				description: 'The user message to answer.',
			},
			mode: {
				type: 'string',
				enum: ['c2c', 'baseline'],
				description: 'c2c (default) fuses the sharer KV-cache; baseline uses the receiver alone.',
			},
			max_new_tokens: {
				type: 'integer',
				description: 'Maximum tokens to generate. Defaults to 256.',
			},
			do_sample: {
				type: 'boolean',
				description:
					'Sample instead of greedy decoding. Defaults to false (deterministic). Keep it false on the Apple MPS backend: torch.multinomial is known to return zero-probability indices there (pytorch#192577), silently corrupting sampled output. Greedy decoding is unaffected.',
			},
			temperature: {
				type: 'number',
				description: 'Sampling temperature; only used when do_sample is true. Avoid on MPS (see do_sample).',
			},
			top_p: {
				type: 'number',
				description: 'Nucleus sampling threshold; only used when do_sample is true. Avoid on MPS (see do_sample).',
			},
			enable_thinking: {
				type: 'boolean',
				description: 'Enable the receiver tokenizer\'s thinking mode. Defaults to false (faster, no reasoning trace).',
			},
		},
		output: {
			schema: { type: 'json' },
			render: (_args, value) => [{ type: 'text', text: renderGeneration(value) }],
		},
		async execute(args) {
			const body = { prompt: args.prompt, mode: args.mode ?? 'c2c' };
			for (const key of ['max_new_tokens', 'do_sample', 'temperature', 'top_p', 'enable_thinking']) {
				if (args[key] !== undefined) body[key] = args[key];
			}
			return client.generate(body);
		},
	}));

	ctx.tools.register(defineTool({
		name: 'c2c_compare',
		description:
			'Run the same prompt twice — once with the sharer KV-cache fused into the receiver (c2c) and once with the receiver alone (baseline) — and return both answers side by side. This is the reproduction check for the Cache-to-Cache paper claim that latent fusion beats single-model answering.',
		parameters: {
			prompt: {
				type: 'string',
				required: true,
				description: 'The user message to answer under both conditions.',
			},
			max_new_tokens: {
				type: 'integer',
				description: 'Maximum tokens to generate per condition. Defaults to 256.',
			},
		},
		output: {
			schema: { type: 'json' },
			render: (_args, value) => [
				{
					type: 'text',
					text: [
						`pair: ${value.pair}    device: ${value.device}`,
						'',
						'=== receiver only (baseline) ===',
						value.baseline.text,
						'',
						'=== C2C fused (sharer KV-cache) ===',
						value.c2c.text,
						'',
						`baseline: ${value.baseline.generated_tokens} tok in ${value.baseline.seconds}s    c2c: ${value.c2c.generated_tokens} tok in ${value.c2c.seconds}s`,
					].join('\n'),
				},
			],
		},
		async execute(args) {
			const common = { prompt: args.prompt };
			if (args.max_new_tokens !== undefined) common.max_new_tokens = args.max_new_tokens;
			const baseline = await client.generate({ ...common, mode: 'baseline' });
			const c2c = await client.generate({ ...common, mode: 'c2c' });
			return { pair: baseline.pair, device: baseline.device, baseline, c2c };
		},
	}));

	ctx.tools.register(defineTool({
		name: 'c2c_unload',
		description:
			'Release the loaded C2C weights and free device memory in the local daemon. The daemon keeps running and can reload a pair on the next call.',
		parameters: {},
		output: {
			schema: { type: 'json' },
			render: (_args, value) => [{ type: 'text', text: JSON.stringify(value, null, 2) }],
		},
		async execute() {
			await client.ensureRunning();
			const result = await client.request('/unload', { method: 'POST', body: {} });
			return result.status ?? result;
		},
	}));

	return client;
}

/** The model-facing usage policy: when and how to drive C2C. */
export function usageSectionText(toolNames) {
	return `Cache-to-Cache (\`c2c_*\` tools): ${toolNames.join(', ')}.
C2C runs two local LLMs where the sharer's KV-cache is projected into the receiver's
cache space and fused before the receiver answers — latent communication instead of
text relay. Use it when the user asks to run, reproduce, or compare Cache-to-Cache,
or to answer a question with a fused model pair.

Workflow: call \`c2c_status\` (or \`c2c_load\` directly) to bring the runtime up, then
\`c2c_chat\`. A cold \`c2c_load\` downloads weights from Hugging Face and can take
minutes; subsequent calls reuse the warm model. Use \`c2c_compare\` when the user
wants to see the fusion benefit against the receiver-only control. The runtime runs
on the local machine (MPS/CUDA/CPU) via a loopback daemon and never leaves it.`;
}

/**
 * Plugin entry point.
 * @param {object} ctx - cordis context with `tools` and `systemPrompt`.
 * @param {object} config - validated plugin config.
 */
export function apply(ctx, config = {}) {
	const resolved = {
		pythonPath: config.pythonPath ?? '',
		repoRoot: config.repoRoot ?? '',
		port: config.port ?? 8765,
		host: config.host ?? '127.0.0.1',
		idleUnloadSeconds: config.idleUnloadSeconds ?? 900,
		loadTimeoutSeconds: config.loadTimeoutSeconds ?? 600,
		requestTimeoutSeconds: config.requestTimeoutSeconds ?? 900,
		logPath: config.logPath ?? '',
		promptSectionOrder: config.promptSectionOrder ?? 118,
	};

	const client = registerTools(ctx, resolved);
	ctx.systemPrompt.section({
		name: 'tool:c2c',
		order: resolved.promptSectionOrder,
		text: usageSectionText(TOOL_NAMES),
	});
	ctx.logger?.info?.(`c2c: registered ${TOOL_NAMES.length} tools (daemon ${client.base})`);
	ctx.on?.('dispose', () => {
		// The daemon is intentionally detached: it outlives this plugin's realm
		// so warm weights survive a profile patch reload.
	});
}
