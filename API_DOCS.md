# TasksAI Privacy Pipeline API Documentation

## Server Setup (RTX 3060 Laptop)

```bash
# Install dependencies
python setup.py

# Start optimized server
python start_server.py
```

Server will be available at: `http://YOUR_LAPTOP_IP:8000`

## Mobile App API Endpoints

### 1. Health Check
```http
GET /api/health
```
**Response:**
```json
{
  "status": "healthy",
  "server": "TasksAI Privacy Pipeline",
  "version": "1.0.0",
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3060",
  "gpu_memory_gb": 12.0,
  "cpu_cores": 8,
  "ram_gb": 16.0
}
```

### 2. System Information
```http
GET /api/system-info
```
**Response:**
```json
{
  "system": {
    "gpu_available": true,
    "gpu_name": "NVIDIA GeForce RTX 3060",
    "optimized_for": "RTX 3060"
  },
  "processing": {
    "batch_size": 8,
    "max_image_size": [2048, 2048],
    "use_gpu_acceleration": true
  },
  "endpoints": {
    "process_single": "/process",
    "process_batch": "/api/process-batch",
    "gallery": "/api/gallery"
  }
}
```

### 3. Process Single File
```http
POST /process
Content-Type: multipart/form-data

file: [image/video file]
```
**Response:**
```json
{
  "filename": "photo.jpg",
  "file_type": "image",
  "privacy_score": 7,
  "is_safe": false,
  "entities": ["EMAIL_ADDRESS", "PHONE_NUMBER"],
  "ocr_text": "Contact: john@email.com, Phone: 555-1234",
  "processing_time": 1.2,
  "result_id": "2025-01-15_photo.jpg"
}
```

### 4. Process Multiple Files (Batch)
```http
POST /api/process-batch
Content-Type: multipart/form-data

files: [multiple image/video files]
```
**Response:**
```json
{
  "processed": 3,
  "results": [
    {
      "filename": "photo1.jpg",
      "privacy_score": 2,
      "is_safe": true,
      "entities": [],
      "processing_time": 0.8
    },
    {
      "filename": "photo2.jpg",
      "privacy_score": 8,
      "is_safe": false,
      "entities": ["EMAIL_ADDRESS", "SSN"],
      "processing_time": 1.1
    }
  ]
}
```

### 5. Get Gallery
```http
GET /api/gallery
```
**Response:**
```json
{
  "items": [
    {
      "id": "2025-01-15_photo.jpg",
      "filename": "photo.jpg",
      "privacy_score": 7,
      "is_safe": false,
      "entities": ["EMAIL_ADDRESS"],
      "file_type": "image",
      "timestamp": "2025-01-15T10:30:00",
      "processing_time": 1.2
    }
  ],
  "total": 1
}
```

### 6. Add Privacy Noise
```http
POST /add-noise?intensity=0.15
Content-Type: multipart/form-data

file: [image file]
```
**Response:** Protected image file download

### 7. Blur Sensitive Areas
```http
POST /blur-sensitive
Content-Type: multipart/form-data

file: [image file]
```
**Response:** Blurred image file download

### 8. Performance Stats
```http
GET /api/stats
```
**Response:**
```json
{
  "uptime_seconds": 3600,
  "total_requests": 150,
  "total_errors": 2,
  "error_rate": 1.33,
  "requests_per_minute": 2.5,
  "processing_time": {
    "avg": 1.2,
    "min": 0.5,
    "max": 3.1,
    "recent_avg": 1.1
  },
  "gpu_usage": {
    "current": 45.2,
    "avg": 38.7,
    "max": 78.3,
    "memory_allocated_mb": 2048,
    "memory_reserved_mb": 2560
  }
}
```

## Privacy Entities Detected

- `PERSON` - Names
- `EMAIL_ADDRESS` - Email addresses
- `PHONE_NUMBER` - Phone numbers
- `CREDIT_CARD` - Credit card numbers
- `SSN` - Social Security Numbers
- `IBAN_CODE` - Bank account numbers
- `IP_ADDRESS` - IP addresses
- `DATE_TIME` - Dates and times
- `LOCATION` - Addresses and locations

## Privacy Score Scale

- **0-3**: Safe (Green) - No privacy concerns
- **4-6**: Low Risk (Yellow) - Minor privacy data detected
- **7-8**: High Risk (Orange) - Significant privacy data
- **9-10**: Critical (Red) - Highly sensitive information

## Error Handling

All endpoints return standard HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid file type, missing file)
- `500`: Server error (processing failed)

Error response format:
```json
{
  "detail": "Error message description"
}
```

## Mobile App Integration Tips

1. **Check server health** before making requests
2. **Use batch processing** for multiple files to improve performance
3. **Monitor performance stats** to optimize user experience
4. **Cache gallery results** and refresh periodically
5. **Handle network errors** gracefully with retry logic
6. **Show processing progress** for better UX

## Network Configuration

- Server runs on port `8000`
- Ensure laptop and mobile device are on same WiFi network
- Configure firewall to allow port `8000` if needed
- Use laptop's local IP address (not localhost) for mobile access