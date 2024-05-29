
# FeTA Challenge 2024

The Fetal Tissue Annotation and Segmentation Challenge (FeTA) at MICCAI 2024 aims to advance multi-class image segmentation and biometric measurement prediction for fetal brain MRIs. The challenge involves developing robust methods using high-resolution, annotated MRI data of fetal brains from gestational weeks 21-36.

## Tasks

### Task 1 - Segmentation

**Training Data:**
- 120 T2-weighted fetal brain reconstructions from two different institutions.
- Each case includes a manually segmented label map with 7 different tissues/labels:
  - External Cerebrospinal Fluid
  - Grey Matter
  - White Matter
  - Ventricles
  - Cerebellum
  - Deep Grey Matter
  - Brainstem
- Data includes clinically acquired reconstructions of both neurotypical and pathological brains across various gestational ages.
- Each case includes gestational age and a neurotypical/pathological label.
- Voxel size: 256x256x256 (resolution varies by institution).

**Testing Data:**
- 180 cases, each with a combined label map and individual label files.
- Expected output: the combined label map.

### Task 2 - Biometry

**Training Data:**
- Biometric measurements for the same 120 subjects as in the segmentation task.
- Five biometry measurements:
  - Brain biparietal diameter (bBIP) in the axial plane
  - Skull biparietal diameter (sBIP) in the axial plane
  - Height of the vermis (HV) in the sagittal plane
  - Length of the corpus callosum (LCC) in the sagittal plane
  - Maximum transverse cerebellar diameter (TCD) in the coronal plane
- Provided data includes a CSV file with target measurements, landmark file in the image space, and a transform to the re-oriented space for measurements.

**Expected Output:**
- CSV file with the predicted length of each structure.
