# LLM 本地配置

项目中的模型信息和 API Key 统一从本地文件读取：`config/llm.local.json`。

## 使用方法

1. 复制模板：

   ```powershell
   Copy-Item config/llm.local.example.json config/llm.local.json
   ```

2. 编辑 `config/llm.local.json`，填写真实的 `api_key`、`model`、`base_url` 等配置。

3. 运行项目时无需再设置 `DOUBAN_API_KEY`、`DOUBAN_MODEL` 等环境变量。

## 字段说明

- `provider`：支持 `douban`、`openai`、`local`。
- `api_key`：远程 API Key。使用 `local` provider 时可留空。
- `model`：模型名称。
- `base_url`：兼容 OpenAI Chat Completions 的远程接口地址。
- `local_url`：本地 llama.cpp 服务地址。
- `timeout`：请求超时时间，单位秒。

`config/llm.local.json` 已加入 `.gitignore`，不会被提交到仓库。
