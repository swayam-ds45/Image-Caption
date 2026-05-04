const express = require('express');
const multer = require('multer');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

// Set up Multer for file uploads
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)){
    fs.mkdirSync(uploadDir);
}

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + path.extname(file.originalname));
    }
});
const upload = multer({ storage: storage });

// Start Python Inference Process
console.log('Starting Python inference process...');
const pythonProcess = spawn('python', ['inference.py']);

let pythonReady = false;
let pendingRequests = {};

// Handle Python process output
pythonProcess.stdout.on('data', (data) => {
    const lines = data.toString().split('\n');
    for (let line of lines) {
        line = line.trim();
        if (!line) continue;
        
        if (line === 'READY') {
            console.log('Python inference script is ready.');
            pythonReady = true;
            continue;
        }

        try {
            const result = JSON.parse(line);
            // Since it's a simple stdin/stdout stream without IDs, 
            // we resolve the oldest pending request.
            // For a robust production app, we would pass IDs back and forth.
            const keys = Object.keys(pendingRequests);
            if (keys.length > 0) {
                const reqId = keys[0];
                const res = pendingRequests[reqId];
                res.json({ success: true, caption: result.caption });
                delete pendingRequests[reqId];
            }
        } catch (e) {
            console.log('Python output:', line);
        }
    }
});

pythonProcess.stderr.on('data', (data) => {
    console.error(`Python Error: ${data}`);
});

pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
    pythonReady = false;
});

// API Endpoint to handle image upload and caption generation
app.post('/api/generate-caption', upload.single('image'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ success: false, message: 'No image uploaded' });
    }

    if (!pythonReady) {
        return res.status(503).json({ success: false, message: 'AI model is still loading. Please try again in a few seconds.' });
    }

    const imagePath = req.file.path;
    const reqId = Date.now().toString();
    
    // Store response object to be resolved when Python script replies
    pendingRequests[reqId] = res;

    // Send image path to Python script
    pythonProcess.stdin.write(imagePath + '\n');
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'ok', pythonReady });
});

app.listen(port, () => {
    console.log(`Backend server running on http://localhost:${port}`);
});
