// Isolated development driver. No Python runtime or input JSON AST in measured RSS.
#include <LightGBM/c_api.h>
#include <dlfcn.h>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

void check(int code) {
  if (code != 0) throw std::runtime_error(LGBM_GetLastError());
}
struct Input {
  std::ifstream stream;
  explicit Input(const char* path) : stream(path, std::ios::binary) {
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    const uint32_t endian = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian) != 1)
      throw std::runtime_error("little-endian host required");
    char magic[8]; stream.read(magic, 8);
    if (std::string(magic, 8) != "LGBMEM01") throw std::runtime_error("invalid input version");
  }
  uint64_t count() {
    uint64_t n; stream.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (n > 2000000) throw std::runtime_error("input count exceeds fixed-case limit");
    return n;
  }
  template<class T> std::vector<T> array() {
    std::vector<T> values(count());
    if (!values.empty()) stream.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(T));
    return values;
  }
};
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b-a).count();
}
void array_json(const char* key, const std::vector<double>& values) {
  std::cout << ",\"" << key << "\":[";
  for (size_t i=0; i<values.size(); ++i) std::cout << (i ? "," : "") << values[i];
  std::cout << "]";
}
int main(int argc, char** argv) {
  try {
    if (argc != 5) throw std::runtime_error("usage: benchmark_memory INPUT_BIN PARAMS_TEXT THREADS PREDICTIONS_BIN");
    Dl_info image{};
    if (!dladdr(reinterpret_cast<void*>(&LGBM_GetLastError), &image))
      throw std::runtime_error("dladdr failed");
    Input input(argv[1]);
    const auto rows=input.count(), cols=input.count(), valid_rows=input.count(), rounds=input.count();
    const bool sparse=input.count() == 1;
    if (rows!=8192 || cols!=16 || valid_rows!=2048 || rounds!=50)
      throw std::runtime_error("not a fixed benchmark case");
    auto train=input.array<double>(), valid=input.array<double>();
    auto labels=input.array<float>(), valid_labels=input.array<float>(), weights=input.array<float>();
    auto groups=input.array<int32_t>();
    auto train_ptr=input.array<int64_t>(); auto train_idx=input.array<int32_t>(); auto train_data=input.array<double>();
    auto valid_ptr=input.array<int64_t>(); auto valid_idx=input.array<int32_t>(); auto valid_data=input.array<double>();
    if (labels.size()!=rows || valid_labels.size()!=valid_rows || input.stream.peek()!=EOF)
      throw std::runtime_error("invalid shape or trailing data");
    input.stream.close();
    std::ifstream param_file(argv[2]);
    std::string params((std::istreambuf_iterator<char>(param_file)), std::istreambuf_iterator<char>());
    // This baseline has USE_OPENMP=OFF. The parameter cannot make it parallel.
    if (std::string(argv[3]) != "1") throw std::runtime_error("pinned native driver only supports actual one thread");
    int classes=params.find("objective=multiclass")!=std::string::npos ? 3 : 1;
    std::vector<double> construction, training, serialization, inference, final_predictions;
    for (int repetition=0; repetition<6; ++repetition) {
      auto start=Clock::now();
      DatasetHandle dataset=nullptr;
      if (sparse) {
        check(LGBM_DatasetCreateFromCSR(train_ptr.data(), C_API_DTYPE_INT64, train_idx.data(),
          train_data.data(), C_API_DTYPE_FLOAT64, train_ptr.size(), train_data.size(), cols,
          params.c_str(), nullptr, &dataset));
      } else {
        check(LGBM_DatasetCreateFromMat(train.data(), C_API_DTYPE_FLOAT64, rows, cols, 1,
          params.c_str(), nullptr, &dataset));
      }
      check(LGBM_DatasetSetField(dataset, "label", labels.data(), labels.size(), C_API_DTYPE_FLOAT32));
      if (!weights.empty()) check(LGBM_DatasetSetField(dataset, "weight", weights.data(), weights.size(), C_API_DTYPE_FLOAT32));
      if (!groups.empty()) check(LGBM_DatasetSetField(dataset, "group", groups.data(), groups.size(), C_API_DTYPE_INT32));
      auto built=Clock::now();
      BoosterHandle booster=nullptr;
      check(LGBM_BoosterCreate(dataset, params.c_str(), &booster));
      for (uint64_t iteration=0; iteration<rounds; ++iteration) {
        int finished; check(LGBM_BoosterUpdateOneIter(booster, &finished));
      }
      auto trained=Clock::now();
      int64_t text_size=0;
      check(LGBM_BoosterSaveModelToString(booster, 0, -1, 0, 0, &text_size, nullptr));
      std::vector<char> text(text_size);
      check(LGBM_BoosterSaveModelToString(booster, 0, -1, 0, text.size(), &text_size, text.data()));
      auto saved=Clock::now();
      std::vector<double> predicted(valid_rows * classes);
      int64_t count=0;
      if (sparse) {
        check(LGBM_BoosterPredictForCSR(booster, valid_ptr.data(), C_API_DTYPE_INT64, valid_idx.data(),
          valid_data.data(), C_API_DTYPE_FLOAT64, valid_ptr.size(), valid_data.size(), cols,
          C_API_PREDICT_NORMAL, 0, -1, "num_threads=1", &count, predicted.data()));
      } else {
        check(LGBM_BoosterPredictForMat(booster, valid.data(), C_API_DTYPE_FLOAT64, valid_rows, cols, 1,
          C_API_PREDICT_NORMAL, 0, -1, "num_threads=1", &count, predicted.data()));
      }
      auto done=Clock::now();
      if (count != static_cast<int64_t>(predicted.size())) throw std::runtime_error("prediction shape mismatch");
      if (repetition) {
        construction.push_back(seconds(start,built)); training.push_back(seconds(built,trained));
        serialization.push_back(seconds(trained,saved)); inference.push_back(seconds(saved,done));
      }
      if (repetition==5) final_predictions=std::move(predicted);
      check(LGBM_BoosterFree(booster)); check(LGBM_DatasetFree(dataset));
    }
    std::ofstream output(argv[4], std::ios::binary);
    output.write(reinterpret_cast<const char*>(final_predictions.data()), final_predictions.size()*sizeof(double));
    if (!output) throw std::runtime_error("prediction write failed");
    std::cout << std::setprecision(17) << "{\"warmup_runs\":1,\"measured_runs\":5,\"threads\":1,\"loaded_library\":\""
      << image.dli_fname << "\",\"prediction_count\":" << final_predictions.size();
    array_json("dataset_construction_seconds",construction); array_json("training_seconds",training);
    array_json("serialization_seconds",serialization); array_json("inference_seconds",inference);
    std::cout << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n"; return 1;
  }
}
