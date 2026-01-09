# Migration Checklist

Use this checklist if you have existing code or scripts that reference the old detector structure.

## ✅ Completed Changes

- [x] Removed `detectors/yolov9/` directory
- [x] Renamed `detectors/ultralytic_yolo/` to `detectors/yolo/`
- [x] Updated training scripts in `scripts/` directory
- [x] Enhanced RF-DETR training and inference scripts
- [x] Created comprehensive documentation
- [x] Updated main README.md

## 📋 Migration Steps for Users

### Step 1: Update Import Paths

If you have Python scripts importing from the old paths:

**Old:**
```python
from detectors.ultralytic_yolo import YOLOv8
from detectors.ultralytic_yolo.utils import load_model
```

**New:**
```python
from detectors.yolo import YOLOv8
from detectors.yolo.utils import load_model
```

### Step 2: Update Script Paths

If you have custom scripts or notebooks:

**Old:**
```bash
python detectors/ultralytic_yolo/train_ultralytics.py --data ...
python detectors/yolov9/train.py --data ...
```

**New:**
```bash
python detectors/yolo/train_ultralytics.py --data ...
# (yolov9 native implementation removed - use ultralytics instead)
```

### Step 3: Update Shell Scripts

If you have custom shell scripts:

**Old:**
```bash
DETECTOR_PATH="detectors/ultralytic_yolo"
YOLOV9_PATH="detectors/yolov9"
```

**New:**
```bash
DETECTOR_PATH="detectors/yolo"
# yolov9 is now included in the unified yolo directory
```

### Step 4: Update Configuration Files

Check any config files (YAML, JSON, etc.) for hardcoded paths:

**Old:**
```yaml
detector_path: detectors/ultralytic_yolo/train_ultralytics.py
```

**New:**
```yaml
detector_path: detectors/yolo/train_ultralytics.py
```

### Step 5: Update Dockerfiles/Docker Compose

If you have containerized deployments:

**Old:**
```dockerfile
COPY detectors/ultralytic_yolo /app/detectors/ultralytic_yolo
COPY detectors/yolov9 /app/detectors/yolov9
```

**New:**
```dockerfile
COPY detectors/yolo /app/detectors/yolo
COPY detectors/rf-detr /app/detectors/rf-detr
```

### Step 6: Update Documentation References

Search your project for references to old paths:

```bash
# Find all references to old paths
grep -r "ultralytic_yolo" .
grep -r "yolov9/train" .

# Update them to new paths
# ultralytic_yolo → yolo
# yolov9/train → yolo (using ultralytics)
```

### Step 7: Test Training Scripts

Verify that training works with the new structure:

```bash
# Test YOLO training
./scripts/train_yolo.sh \
  --data data/sample/data.yaml \
  --model yolov8n.yaml \
  --epochs 1 \
  --batch 4

# Test RF-DETR training
python detectors/rf-detr/train.py \
  --data data/sample \
  --epochs 1 \
  --batch 4
```

### Step 8: Update Model Loading Code

If you have inference code loading models:

**Old:**
```python
model_path = "detectors/ultralytic_yolo/yolov9-m-converted.pt"
```

**New:**
```python
model_path = "detectors/yolo/yolov9-m-converted.pt"
```

### Step 9: Update CI/CD Pipelines

Check your CI/CD configuration files:

**GitHub Actions:**
```yaml
# Old
- name: Train model
  run: python detectors/ultralytic_yolo/train_ultralytics.py ...

# New
- name: Train model
  run: python detectors/yolo/train_ultralytics.py ...
```

**GitLab CI:**
```yaml
# Old
script:
  - python detectors/ultralytic_yolo/train_ultralytics.py ...

# New
script:
  - python detectors/yolo/train_ultralytics.py ...
```

### Step 10: Update README/Documentation

Update any project-specific documentation:

1. Replace references to `ultralytic_yolo` with `yolo`
2. Remove references to `yolov9` native implementation
3. Add references to new RF-DETR documentation
4. Update training examples with new paths

## 🔍 Verification Commands

After migration, verify everything works:

```bash
# 1. Check directory structure
ls -la detectors/
# Should show: yolo/, rf-detr/, README.md

# 2. Verify Python imports
python -c "from detectors.yolo.train_ultralytics import main; print('✓ YOLO imports OK')"

# 3. Check training script
./scripts/train_yolo.sh --help

# 4. Verify RF-DETR
python detectors/rf-detr/train.py --help

# 5. Run a quick test
./scripts/train_yolo.sh \
  --data data/test/data.yaml \
  --model yolov8n.yaml \
  --epochs 1 \
  --batch 2
```

## 🐛 Common Issues

### Issue 1: Import Error
```
ModuleNotFoundError: No module named 'detectors.ultralytic_yolo'
```

**Solution:** Update import to `from detectors.yolo import ...`

### Issue 2: File Not Found
```
FileNotFoundError: detectors/ultralytic_yolo/train_ultralytics.py
```

**Solution:** Update path to `detectors/yolo/train_ultralytics.py`

### Issue 3: YOLOv9 Training Script Missing
```
FileNotFoundError: detectors/yolov9/train.py
```

**Solution:** Use Ultralytics YOLO instead:
```bash
python detectors/yolo/train_ultralytics.py --model yolov9m.yaml ...
```

### Issue 4: Old Model Paths
```
Model file not found: detectors/ultralytic_yolo/yolov9-s.pt
```

**Solution:** Update to new path:
```python
model_path = "detectors/yolo/yolov9-s.pt"
```

## 📚 Resources

- **Main README**: [README.md](README.md)
- **Detectors Overview**: [detectors/README.md](detectors/README.md)
- **RF-DETR Guide**: [detectors/rf-detr/README.md](detectors/rf-detr/README.md)
- **Training Quick Reference**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Refactoring Summary**: [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)

## ✉️ Support

If you encounter issues during migration:

1. Check this checklist thoroughly
2. Review error messages carefully
3. Verify all paths have been updated
4. Check that dependencies are installed
5. Refer to the documentation above

## 🎯 Next Steps

After completing migration:

1. ✅ Commit your changes
2. ✅ Update team documentation
3. ✅ Notify team members of changes
4. ✅ Test in development environment
5. ✅ Deploy to production after verification

---

**Migration completed?** Run a full test suite to ensure everything works correctly!
