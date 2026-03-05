# FaceRecognition (KP-RPE 실험용)

간단 사용 문서입니다.  
핵심만 적었습니다.

## 1) 현재 상태
- `lfw`, `agedb_30`, `cfp_fp`는 `.bin -> run_v1 eval 포맷` 변환 가능 상태
- `cplfw`, `calfw`는 아직 `.bin` 파일이 없음
- `tinyface`, `IJB-B/C`는 raw 데이터는 있어도 `run_v1` 평가 포맷(`datasets.load_from_disk`)으로 아직 미변환

## 2) 학습 데이터 전처리 (MTCNN 5-point align)
`run_v1` 방식처럼 bbox crop이 아니라 **landmark 기반 similarity align**으로 저장합니다.

### VGGFace2
```bash
bash shells/preprocessing.sh \
  vgg2 \
  /data/mj/vgg2 \
  /data/mj/vgg2_aligned \
  cuda:0 32 4
```

### CASIA parquet(raw)
```bash
bash shells/preprocessing.sh \
  casia \
  /data/mj/casia-webface-hf \
  /data/mj/casia-webface-aligned \
  cuda:0 32 4
```

전처리 진행도:
- tqdm progress bar 표시
- non-tty 환경에서도 `--log_interval`마다 진행 로그 출력

직접 실행 시:
```bash
python preprocessing.py \
  --dataset_name casia \
  --input_root /data/mj/casia-webface-hf \
  --output_root /data/mj/casia-webface-aligned \
  --device cuda:0 \
  --batch_size 32 \
  --num_workers 4 \
  --log_interval 200
```

## 3) run_v1 검증셋 준비
`run_v1` 검증 코드는 HuggingFace `datasets.load_from_disk` 포맷을 기대합니다.

### 의존성
```bash
pip install -U datasets pyarrow
```

### (A) 현재 있는 verification bin 변환
```bash
python tools/prepare_verification_eval.py \
  --bin_root /data/mj/eval_bins \
  --out_root /data/mj/facerec_val \
  --names lfw agedb_30 cfp_fp
```

### (B) 남은 cplfw/calfw 다운로드 후 변환
```bash
hf download namkuner/namkuner_face_dataset \
  --repo-type dataset \
  cplfw.bin calfw.bin \
  --local-dir /data/mj/eval_bins

python tools/prepare_verification_eval.py \
  --bin_root /data/mj/eval_bins \
  --out_root /data/mj/facerec_val \
  --names cplfw calfw
```

### (C) 준비 상태 확인
```bash
python tools/check_eval_ready.py --root /data/mj
```

### (D) TinyFace / IJB-C / IJB-S 포함 전체 파이프라인(다운로드+전처리)
```bash
bash shells/prepare_eval_pipeline.sh /data/mj cuda:0 256 4
```

추가된 스크립트:
- `tools/prepare_tinyface_eval.py`: TinyFace -> `facerec_val/tinyface_aligned_pad_0.1`
- `tools/prepare_ijbc_eval.py`: IJB-C(옵션으로 IJB-B) -> `facerec_val/IJBC_gt_aligned`
- `tools/prepare_ijbs_aligned.py`: IJB-S 이미지 정렬(현재 run_v1엔 IJB-S evaluator 없음)

## 4) dataset_name 요약
- MS1M 계열: `ms1mv3`
- VGG 계열: `vgg2`, `vgg2_raw`, `vgg2_aligned`
- CASIA aligned 폴더: `casia`, `casia_aligned`
- CASIA raw parquet: `casia_raw`, `casia_parquet`

## 5) 참고
- `tinyface`, `IJB-B/C`는 현재 repo에 변환 스크립트가 아직 없음
- raw 데이터만으로는 `run_v1` eval에 바로 들어가지 않음 (`Dataset.load_from_disk` + `metadata.pt` 필요)
