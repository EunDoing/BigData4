# MovieLens 1M Streaming Algorithm Benchmark

대용량 데이터 스트림 환경에서 전체 데이터를 저장하지 않고 근사 계산을 수행하는 확률적 자료구조(Probabilistic Data Structures)인 **Bloom Filter**와 **Count-Min Sketch**를 직접 구현하고 성능을 비교 분석한 실험 프로젝트입니다.

---

## 프로젝트 개요
* **과제명:** 스트리밍 알고리즘 2종 구현 및 정확도·메모리 트레이드오프 분석
* **분석 데이터셋:** MovieLens 1M (`ratings.dat`, 총 1,000,209건의 평점 로그)
* **핵심 설계:** - 대용량 시나리오를 모사하기 위해 Python Generator(`yield`)를 활용한 **Single-pass(1회 통과) 스트림 처리**
  - 단 한 번의 스트림 통과로 **모든 파라미터 조합(Multi-config)을 동시에 벤치마킹**하여 디스크 I/O 최적화

---

## 🛠️ 개발 및 실험 환경
* **Language:** Python 3.12
* **Libraries:** `mmh3` (MurmurHash3), `numpy`, `matplotlib`, `tracemalloc`
* **Dataset Structure:** `UserID::MovieID::Rating::Timestamp`

---
