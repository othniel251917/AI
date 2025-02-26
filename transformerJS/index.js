import express from "express";
import multer from "multer";
import path from "path";
import { readFile } from 'fs/promises';
import { RawImage } from '@xenova/transformers';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const port = 3000;

// 配置文件上传
const storage = multer.diskStorage({
  destination: "./uploads/",
  filename: function (req, file, cb) {
    cb(
      null,
      file.fieldname + "-" + Date.now() + path.extname(file.originalname)
    );
  },
});

const upload = multer({ storage: storage });

// 服务静态文件
app.use(express.static("public"));

const initModel = async () => {
  const { AutoModelForCausalLM, AutoProcessor } = await import("@xenova/transformers");
  
  // 并行加载模型和处理器
  const [model, processor] = await Promise.all([
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-VL-3B-Instruct", {
      quantized: false, // 如果遇到内存问题可以设置为true
      device: "cpu",    // 根据环境切换 "cuda" 或 "cpu"
    }),
    AutoProcessor.from_pretrained("Qwen/Qwen1.5-VL-3B-Instruct")
  ]);

  return { model, processor };
};

// 处理图片上传和模型推理
app.post("/analyze", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "没有上传图片" });
    }

    const question = req.body.question || "请描述这张图片";
    const result = await chatWithImage(req.file.path, question);

    res.json({ result });
  } catch (error) {
    console.error("处理错误:", error);
    res.status(500).json({ error: "处理图片时发生错误" });
  }
});

// 图像对话函数
async function chatWithImage(imagePath, question) {
  try {
    const { model, processor } = await initModel();
    
    // 1. 读取并预处理图像
    const imageBuffer = await readFile(imagePath);
    const image = await RawImage.fromBuffer(imageBuffer);
    
    // 2. 构造符合模型要求的对话模板
    const prompt = [
      { role: "user", content: [
        { type: "image" }, // 图像占位符
        { type: "text", text: question }
      ]}
    ];
    
    // 3. 处理多模态输入
    const processed = await processor(prompt, image);
    
    // 4. 生成响应
    const outputs = await model.generate({
      ...processed,
      max_new_tokens: 500,
      pad_token_id: processor.tokenizer.pad_token_id,
    });
    
    // 5. 解码输出
    return processor.decode(outputs[0], { 
      skip_special_tokens: true 
    });
  } catch (error) {
    console.error("处理失败:", error);
    throw error;
  }
}

app.listen(port, () => {
  console.log(`服务器运行在 http://localhost:${port}`);
});
