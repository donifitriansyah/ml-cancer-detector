const express = require("express");
const multer = require("multer");
const { Storage } = require("@google-cloud/storage");
const tf = require("@tensorflow/tfjs-node");
const { Firestore } = require("@google-cloud/firestore");
const { v4: uuidv4 } = require("uuid");

const app = express();
const firestore = new Firestore();
const storage = new Storage();
const bucketName = "ml-model-bucket-dicoding"; 
const modelPath = "models/model.json"; 

let model;

const upload = multer({
  limits: { fileSize: 1 * 1024 * 1024 }, 
}).single("image");

async function loadModel() {
  const [files] = await storage.bucket(bucketName).getFiles({ prefix: modelPath });
  if (!files.length) {
    throw new Error("Model not found in the specified bucket path.");
  }
  const modelUrl = `gs://${bucketName}/${modelPath}`;
  model = await tf.loadGraphModel(modelUrl);
}

// Endpoint untuk prediksi
app.post("/predict", upload, async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ status: "fail", message: "Image is required" });
    }

    const imageBuffer = req.file.buffer;
    const tensor = tf.node.decodeImage(imageBuffer, 3).resizeNearestNeighbor([224, 224]).expandDims(0);
    const prediction = model.predict(tensor);
    const predictionValue = (await prediction.data())[0];

    const id = uuidv4();
    const isCancer = predictionValue > 0.5;
    const result = isCancer ? "Cancer" : "Non-cancer";
    const suggestion = isCancer ? "Segera periksa ke dokter!" : "Penyakit kanker tidak terdeteksi.";
    const createdAt = new Date().toISOString();

    // Simpan hasil ke Firestore
    await firestore.collection("predictions").doc(id).set({
      id,
      result,
      suggestion,
      createdAt,
    });

    res.json({
      status: "success",
      message: "Model is predicted successfully",
      data: {
        id,
        result,
        suggestion,
        createdAt,
      },
    });
  } catch (error) {
    if (error.message.includes("File too large")) {
      return res.status(413).json({
        status: "fail",
        message: "Payload content length greater than maximum allowed: 1000000",
      });
    }
    res.status(400).json({
      status: "fail",
      message: "Terjadi kesalahan dalam melakukan prediksi",
    });
  }
});

// Endpoint untuk riwayat prediksi
app.get("/predict/histories", async (req, res) => {
  try {
    const snapshot = await firestore.collection("predictions").get();
    const data = snapshot.docs.map(doc => ({
      id: doc.id,
      history: doc.data(),
    }));
    res.json({ status: "success", data });
  } catch (error) {
    res.status(500).json({ status: "fail", message: "Failed to fetch prediction history" });
  }
});

// Mulai server dan load model
const PORT = process.env.PORT || 3000;
app.listen(PORT, async () => {
  try {
    await loadModel();
    console.log(`Server is running on http://localhost:${PORT}`);
  } catch (error) {
    console.error("Error loading model from GCS:", error);
  }
});
