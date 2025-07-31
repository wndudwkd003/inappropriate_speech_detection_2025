## 소개
2025년 국립국어원 AI말평 경진대회 참여를 위해 Unsloth 라이브러리를 활용하여 LLM 개발을 진행하였습니다.


## 사용방법
1. 레퍼지토리 클론
```
git clone https://github.com/wndudwkd003/kr_llm_ft_25.git
cd kr_llm_ft_25
```

2. 학습 실행 쉘
```
. run/run_sft_train.sh
```

3. 평가 실행 쉘 *(내부 쉘 파일의 경로 설정 주의)*
```
. run/run_sft_test.sh
```

### 주의사항
1. 학습과 평가를 진행할 때 파라미터의 데이터 경로를 정확하게 설정한다.
2. 평가를 진행할 때 학습 시 저장되는 폴더의 경로를 정확하게 설정한다.

### 파라미터 수정 방법
- configs의 각종 config를 변경하면 된다.


## 검색 증강
### 옵션 설정
1. configs/rag_config.yaml에서 source_files을 data/rag/corpus.pdf로 설정한다.
2. output_dir를 data/rag_results로 설정한다.
3. index_dir를 data/rag/index로 설정한다.
4. 마지막으로 use_rag를 true로 설정한다.

### 실행 순서
1. 문서로부터 인덱스와 메타 데이터를 생성
```
run/run_build_index.sh
```

2. 생성된 인덱스와 메타데이터로부터 실제 데이터 세트를 검색 증강
```
run/run_augmentation_rag.sh
```


## 도커 환경에서 실행하기

### 도커 이미지 불러오기
```
비공개 (내부 주요 파일 수정 후 공개 예정)
```


### 도커 컨테이너 생성

- 마운트 환경 생각해서 실행 명령어 입력
```
docker run -it --name kjy_kli_llm --gpus all ymail3/kr_llm_ft_25:v_1 /bin/bash
```

- 예시
```
docker run -it --name kjy_kli_llm \
  -v /dev/hdd/user/kjy/kli_llm_workspace:/workspace \
  -v /home/oem/.cache/huggingface:/root/.cache/huggingface \
  --gpus all \
  ymail3/kr_llm_ft_25:v_1 \
  /bin/bash
```

### 도커 컨테이너 입장
```
docker exec -it <컨테이너ID> bash
```



### 폴더 구조

```
.
|-- configs
|   |-- dpo_config.yaml
|   |-- lora_config.yaml
|   |-- model_config.yaml
|   |-- sft_config.yaml
|   `-- system_config.yaml
|-- data
|   |-- rag
|   |   |-- corpus.pdf
|   |   `-- index
|   `-- raw
|       |-- dev.json
|       |-- test.json
|       `-- train.json
|-- output
|   `-- 2025-07-24_07-40-50_kakaocorp_kanana-1.5-8b-base_r_128_ra_256_rd_0_sft
|       |-- README.md
|       |-- checkpoint-1200
|       |   |-- README.md
|       |   |-- adapter_config.json
|       |   |-- adapter_model.safetensors
|       |   |-- chat_template.jinja
|       |   |-- optimizer.pt
|       |   |-- rng_state.pth
|       |   |-- scaler.pt
|       |   |-- scheduler.pt
|       |   |-- special_tokens_map.json
|       |   |-- tokenizer.json
|       |   |-- tokenizer_config.json
|       |   |-- trainer_state.json
|       |   `-- training_args.bin
|       |-- configs
|       |   |-- lora_config.yaml
|       |   |-- model_config.yaml
|       |   |-- sft_config.yaml
|       |   `-- system_config.yaml
|       |-- logs
|       |   `-- events.out.tfevents.1753342868.c0e82bb3dc9c.2233602.0
|       |-- lora_adapter
|       |   |-- README.md
|       |   |-- adapter_config.json
|       |   `-- adapter_model.safetensors
|       `-- metrics.json
|-- readme.md
|-- run
|   `-- run_sft_train.sh
|-- scripts
|   |-- __pycache__
|   |   |-- train_sft.cpython-311.pyc
|   |   `-- train_sft.cpython-312.pyc
|   |-- train_dpo.py
|   `-- train_sft.py
|-- src
|   |-- configs
|   |   |-- __pycache__
|   |   |   |-- config_manager.cpython-312.pyc
|   |   |   |-- lora_config.cpython-312.pyc
|   |   |   |-- model_config.cpython-312.pyc
|   |   |   |-- sft_config.cpython-312.pyc
|   |   |   `-- system_config.cpython-312.pyc
|   |   |-- config_manager.py
|   |   |-- lora_config.py
|   |   |-- model_config.py
|   |   |-- sft_config.py
|   |   |-- system_config.py
|   |   `-- tokens
|   |       `-- token.yaml
|   |-- data
|   |   |-- __pycache__
|   |   |   |-- dataset.cpython-312.pyc
|   |   |   `-- sft_dataset.cpython-312.pyc
|   |   |-- dataset.py
|   |   |-- dpo_dataset.py
|   |   `-- sft_dataset.py
|   |-- models
|   |-- rag
|   |   |-- indexer.py
|   |   |-- rag_dataset.py
|   |   `-- retriever.py
|   |-- test
|   |-- train
|   |   |-- __pycache__
|   |   |   `-- sft_trainer.cpython-312.pyc
|   |   |-- dpo_trainer.py
|   |   `-- sft_trainer.py
|   `-- utils
|       |-- __pycache__
|       |   |-- data_utils.cpython-312.pyc
|       |   `-- path_utils.cpython-312.pyc
|       |-- adapter_utils.py
|       |-- data_utils.py
|       `-- path_utils.py
```
