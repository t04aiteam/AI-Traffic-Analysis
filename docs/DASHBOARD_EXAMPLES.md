# Dashboard Usage Examples

## Example 1: Training a License Plate Detector with Roboflow

### Scenario
You want to train a custom license plate detector using your Roboflow dataset.

### Steps

1. **Access the Dashboard**
   ```
   http://localhost:8000/dashboard.html
   ```

2. **Navigate to Training Tab**
   - Click on the "Training" tab

3. **Configure Job**
   - Job Name: `vietnam_plates_v1`
   - Model Architecture: Select `YOLOv8 Nano` (fastest, good for testing)

4. **Set Up Dataset (Roboflow)**
   - Keep "Roboflow" tab selected
   - API Key: Get from https://app.roboflow.com/settings/api
   - Workspace: `your-workspace-name`
   - Project: `license-plate-detection`
   - Version: `1`

5. **Configure Training Parameters**
   - Epochs: `100` (default)
   - Batch Size: `16` (adjust based on GPU memory)
   - Image Size: `640`
   - Device: `GPU 0`
   - Workers: `8`

6. **Start Training**
   - Click "Start Training"
   - You'll see confirmation with Job ID

7. **Monitor Progress**
   - Automatically redirected to "Training Jobs" tab
   - Click on your job to see detailed progress
   - Watch real-time logs

8. **After Completion**
   - Find trained model at: `runs/detect/vietnam_plates_v1_YYYYMMDD_HHMMSS/weights/best.pt`
   - Copy to weights folder for use in production

---

## Example 2: Training with Local Dataset

### Scenario
You already have a dataset prepared locally with a `data.yaml` file.

### Steps

1. **Prepare Dataset Structure**
   ```
   data/my_dataset/
   ├── data.yaml
   ├── train/
   │   ├── images/
   │   └── labels/
   └── val/
       ├── images/
       └── labels/
   ```

2. **Create data.yaml**
   ```yaml
   path: /root/tungn197/license-plate-recognition/data/my_dataset
   train: train/images
   val: val/images
   
   names:
     0: license_plate
   ```

3. **Configure Training**
   - Job Name: `custom_detector_v1`
   - Model: `YOLO11 Nano`
   - Select "Local data.yaml" tab
   - Path: `data/my_dataset/data.yaml` (relative to project root)

4. **Adjust Parameters**
   - Epochs: `200` (more epochs for better results)
   - Batch Size: `32` (if you have good GPU)
   - Image Size: `640`

5. **Start and Monitor**
   - Click "Start Training"
   - Monitor in Training Jobs tab

---

## Example 3: Managing Models

### Scenario
You need to organize and download trained models.

### Browse Models
1. Go to "Model Library" tab
2. See all models organized by category
3. Use search: type "vehicle" to find vehicle detectors
4. Use filter: select "Plate" category

### Download a Model
1. Find the model you want
2. Click the download icon (⬇)
3. Model file downloads to your computer
4. Use it in your application

### Delete Old Models
1. Find the model to remove
2. Click the trash icon (🗑)
3. Confirm deletion
4. Model is removed from weights folder

---

## Example 4: System Monitoring

### Scenario
Check if your system can handle training before starting.

### Check GPU
1. Go to "System Info" tab
2. Look at GPU Information section
3. Verify:
   - GPU is available
   - Sufficient free memory (>8GB recommended)
   - Current utilization is low

### Check Storage
1. Look at Storage Information
2. Verify:
   - Workspace has enough free space (>20GB recommended)
   - Training Runs folder isn't too large
   - Model Weights folder size

### Monitor During Training
1. Keep System Info tab open
2. Click "Refresh" periodically
3. Watch:
   - GPU utilization increasing
   - Memory usage
   - Storage consumption

---

## Example 5: Handling Failed Training

### Scenario
A training job failed and you need to troubleshoot.

### Identify the Problem
1. Go to "Training Jobs" tab
2. Click on the failed job (red border)
3. Read the error message
4. Check the logs for details

### Common Issues and Solutions

#### Out of Memory
```
Error: CUDA out of memory
```
**Solution**: Reduce batch size
- Try batch size: 8 → 4 → 2
- Or use smaller model (nano instead of large)
- Or reduce image size: 640 → 416

#### Dataset Not Found
```
Error: data.yaml not found
```
**Solution**: Check path
- Use absolute path
- Or verify relative path from project root
- Ensure data.yaml exists

#### Invalid Model
```
Error: Model not found: yolov8x.yaml
```
**Solution**: Use available model
- Check available models in dropdown
- Use exact model name from list

### Retry Training
1. Note the issue
2. Go to "Training" tab
3. Adjust parameters based on error
4. Start new training job

---

## Example 6: Comparing Training Results

### Scenario
You trained multiple models and want to compare them.

### Train Variations
1. **First Training**
   - Name: `plate_detector_small_batch16`
   - Model: YOLO11 Small
   - Batch: 16
   - Epochs: 100

2. **Second Training**
   - Name: `plate_detector_small_batch32`
   - Model: YOLO11 Small
   - Batch: 32
   - Epochs: 100

3. **Third Training**
   - Name: `plate_detector_nano_batch16`
   - Model: YOLO11 Nano
   - Batch: 16
   - Epochs: 100

### Compare Results
1. Go to "Training Jobs" tab
2. See all completed jobs
3. Click each job to:
   - View final loss values in logs
   - Check training time
   - Download logs for detailed analysis

4. Test each model:
   - Find best.pt in runs/detect/{job_name}/weights/
   - Test on validation set
   - Compare accuracy and speed

---

## Example 7: Batch Training Multiple Models

### Scenario
You want to train different model sizes overnight.

### Queue Multiple Jobs
1. **Job 1**: Nano model (fast)
   - Configure and start
   - Takes ~2 hours

2. **Job 2**: Small model (medium)
   - Configure and start immediately after
   - Takes ~4 hours

3. **Job 3**: Medium model (slow)
   - Configure and start after
   - Takes ~8 hours

**Note**: Jobs run sequentially (one at a time) to avoid GPU conflicts.

### Morning Review
1. Check "Training Jobs" tab
2. See which jobs completed
3. Download all logs
4. Test all trained models

---

## Example 8: Using Custom Pretrained Weights

### Scenario
You want to fine-tune an existing model instead of training from scratch.

### Prepare
1. Place pretrained weights in `weights/pretrained/`
2. Example: `yolov8s_pretrained.pt`

### Configure Training
1. Model Type: Select base architecture
2. In Training Parameters section (would need to add this feature):
   - Pretrained Weights: Browse to your .pt file
   - Or use --weights flag in manual training

**Current Workaround**:
1. Modify model name to point to .pt file
2. Instead of "yolov8s.yaml", use "yolov8s.pt"
3. Or place .pt file in root and reference it

---

## Example 9: Real-Time Monitoring While Training

### Scenario
Monitor your training job as it progresses.

### Setup Multi-View
1. Open dashboard in browser
2. Click on training job to open modal
3. Keep modal open
4. Observe:
   - Progress bar advancing
   - Epoch counter increasing
   - Logs appearing in real-time
   - Loss values decreasing (hopefully!)

### Mobile Monitoring
1. Open dashboard on phone: `http://your-server-ip:8000/dashboard.html`
2. Responsive design works on mobile
3. Check progress anywhere

---

## Example 10: Exporting and Sharing Results

### Scenario
Training completed and you want to share results with team.

### Export Logs
1. Open completed job
2. Click "Download Logs"
3. Save as `job_name_logs.txt`
4. Share via email/chat

### Export Model
1. Go to "Model Library"
2. Navigate to training runs (if you copied model there)
3. Or manually:
   ```bash
   cp runs/detect/job_name/weights/best.pt \
      weights/plate/custom_plate_detector_v1.pt
   ```

### Create Report
1. Download logs
2. Note training parameters
3. Document:
   - Dataset size
   - Model architecture
   - Training time
   - Final metrics
   - Best weight location

---

## Tips & Tricks 💡

### Performance
- Use smaller batch size if GPU memory is limited
- Start with nano models for quick experiments
- Use pretrained weights for faster convergence

### Organization
- Use descriptive job names with version numbers
- Include date in job names: `detector_v1_20240109`
- Keep old models in archive folder

### Monitoring
- Check system info before starting long training
- Set up email notifications (future feature)
- Use screen/tmux if SSH connection might drop

### Troubleshooting
- Always check logs first
- Reduce batch size for memory errors
- Verify dataset format matches YOLO requirements
- Test with small epochs (5-10) before full training

### Best Practices
- Start with nano model for quick validation
- Use 10-20% of epochs for testing
- Monitor first few epochs closely
- Keep multiple checkpoints
- Document what worked and what didn't
