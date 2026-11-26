# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.1] - 2026-11-26

### Added

- Tells toptic.ai where the context window reported for each Ollama model comes from
  (`context_length_source`: `configured`, `loaded`, `modelfile` or `default`). The `default` of
  4096 is a guess, since the server's `OLLAMA_CONTEXT_LENGTH` is visible only once a model is
  loaded: toptic.ai keeps the window it last saw for a loaded model while that model is out of
  memory, instead of falling back to 4096, and still follows a `context_length` or `num_ctx`
  that was changed or removed.
- When a model of an Ollama backend without `context_length` comes into memory, or Ollama loads it
  again with another window, a fresh models report follows soon (within toptic.ai's limit of one a
  minute), not up to five minutes later, so toptic.ai learns its real window promptly. Needs a
  toptic.ai that is told what is in memory. After lowering `OLLAMA_CONTEXT_LENGTH` (or creating a
  model again with a smaller `num_ctx`), toptic.ai keeps planning with the window last seen loaded
  until the model has been loaded again, usually about a minute after the next request; setting
  `context_length` for the backend pins the window.

### Fixed

- An Ollama model whose details (`/api/show`) could not be read was reported without a context
  window, also where `context_length` was configured or the model was loaded; it now gets the
  window from those, else Ollama's default.

## [0.1.0] - 2026-11-22

### Added

- Makes the models on your own machine usable in toptic.ai: Ollama, a LiteLLM proxy (local or
  remote) or any other OpenAI-compatible server. The client dials out to toptic.ai over a
  WebSocket, so the machine needs no public address and nothing listens on it.
- `socketllama setup` asks for the connection and the models to share and writes `config.toml`
  for its owner only; `socketllama check` validates the configuration and lists the models of
  each backend; `socketllama run` connects and relays requests, reconnecting by itself.
- Relays only signed `POST /chat/completions` requests for the models you share, to the backends
  you configured, and streams the answers back. Backend API keys stay on the machine, and neither
  secrets nor the text of a prompt or an answer are ever logged.
- Reports the shared models to toptic.ai: of Ollama with the context window requests really get,
  of a LiteLLM proxy with its limits and prices.
- For Ollama: keeps models in memory longer (`keep_loaded`) or loads them at start
  (`load_at_start`) if asked to, waits as long as toptic.ai does for a model to load, finishes a
  load whose request was given up, and tells toptic.ai which shared models are in memory.
- A price and a description per model, set in `config.toml` or on the machine's page in
  toptic.ai, kept the same in both places.
- One line per event in the terminal: the connection, the models reported, each prompt with who
  sent it and its size, and how its answer ended.

[0.1.1]: https://github.com/toptic-ai/SocketLlama/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/toptic-ai/SocketLlama/releases/tag/v0.1.0
