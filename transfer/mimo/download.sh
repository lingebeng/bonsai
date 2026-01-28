# 创建并进入文件夹 (保持整洁)
mkdir -p MiMo-V2-Flash
cd MiMo-V2-Flash

# 1. 下载配置文件与 Tokenizer (小文件)
wget -c https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/resolve/main/config.json
wget -c https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/resolve/main/added_tokens.json
wget -c https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/resolve/main/special_tokens_map.json
wget -c https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/resolve/main/tokenizer_config.json
wget -c https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/resolve/main/tokenizer.json
wget -c https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/resolve/main/vocab.json
wget -c https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/resolve/main/model.safetensors.index.json

# 2. 下载模型权重 (大文件)
wget -c https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/resolve/main/model_0.safetensors
wget -c https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/resolve/main/model_1.safetensors
wget -c https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/resolve/main/model_embedding.safetensors
wget -c https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/resolve/main/model_final.safetensors
wget -c https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/resolve/main/model_mtp.safetensors

# 3. 下载特定的 Linear 层权重 (根据你提供的列表)
wget -c https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/resolve/main/model_1_linear_fc1.safetensors
wget -c https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/resolve/main/model_1_linear_fc2.safetensors

echo "所有文件下载完成！"