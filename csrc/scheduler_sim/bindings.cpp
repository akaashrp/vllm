#include <pybind11/pybind11.h>

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace vllm::scheduler_sim {

namespace {

struct MsgpackValue {
  enum class Kind { Null, Bool, Int, UInt, Double, String, Array, Map };

  Kind kind = Kind::Null;
  bool bool_value = false;
  int64_t int_value = 0;
  uint64_t uint_value = 0;
  double double_value = 0.0;
  std::string string_value;
  std::vector<MsgpackValue> array_value;
  std::vector<std::pair<std::string, MsgpackValue>> map_value;

  static MsgpackValue Null() { return {}; }
  static MsgpackValue FromBool(bool value) {
    MsgpackValue out;
    out.kind = Kind::Bool;
    out.bool_value = value;
    return out;
  }
  static MsgpackValue FromInt(int64_t value) {
    MsgpackValue out;
    out.kind = Kind::Int;
    out.int_value = value;
    return out;
  }
  static MsgpackValue FromUInt(uint64_t value) {
    MsgpackValue out;
    out.kind = Kind::UInt;
    out.uint_value = value;
    return out;
  }
  static MsgpackValue FromDouble(double value) {
    MsgpackValue out;
    out.kind = Kind::Double;
    out.double_value = value;
    return out;
  }
  static MsgpackValue FromString(std::string value) {
    MsgpackValue out;
    out.kind = Kind::String;
    out.string_value = std::move(value);
    return out;
  }
  static MsgpackValue FromArray(std::vector<MsgpackValue> values) {
    MsgpackValue out;
    out.kind = Kind::Array;
    out.array_value = std::move(values);
    return out;
  }
  static MsgpackValue FromMap(
      std::vector<std::pair<std::string, MsgpackValue>> entries) {
    MsgpackValue out;
    out.kind = Kind::Map;
    out.map_value = std::move(entries);
    return out;
  }
};

class MsgpackParser {
 public:
  MsgpackParser(const char* data, size_t length)
      : data_(reinterpret_cast<const uint8_t*>(data)), size_(length) {}

  MsgpackValue parse() {
    MsgpackValue value = parse_value();
    if (offset_ != size_) {
      throw std::runtime_error("Unexpected bytes after msgpack payload");
    }
    return value;
  }

 private:
  MsgpackValue parse_value() {
    uint8_t prefix = read_byte();
    if (prefix <= 0x7f) {
      return MsgpackValue::FromUInt(prefix);
    }
    if (prefix >= 0xe0) {
      return MsgpackValue::FromInt(static_cast<int8_t>(prefix));
    }
    if ((prefix & 0xe0) == 0xa0) {
      const size_t size = prefix & 0x1f;
      return MsgpackValue::FromString(read_string(size));
    }
    if ((prefix & 0xf0) == 0x80) {
      const size_t size = prefix & 0x0f;
      return parse_map(size);
    }
    if ((prefix & 0xf0) == 0x90) {
      const size_t size = prefix & 0x0f;
      return parse_array(size);
    }

    switch (prefix) {
      case 0xc0:
        return MsgpackValue::Null();
      case 0xc2:
        return MsgpackValue::FromBool(false);
      case 0xc3:
        return MsgpackValue::FromBool(true);
      case 0xcc:
        return MsgpackValue::FromUInt(read_unsigned(1));
      case 0xcd:
        return MsgpackValue::FromUInt(read_unsigned(2));
      case 0xce:
        return MsgpackValue::FromUInt(read_unsigned(4));
      case 0xcf:
        return MsgpackValue::FromUInt(read_unsigned(8));
      case 0xd0:
        return MsgpackValue::FromInt(read_signed(1));
      case 0xd1:
        return MsgpackValue::FromInt(read_signed(2));
      case 0xd2:
        return MsgpackValue::FromInt(read_signed(4));
      case 0xd3:
        return MsgpackValue::FromInt(read_signed(8));
      case 0xca:
        return MsgpackValue::FromDouble(read_float32());
      case 0xcb:
        return MsgpackValue::FromDouble(read_float64());
      case 0xd9:
        return MsgpackValue::FromString(read_string(read_unsigned(1)));
      case 0xda:
        return MsgpackValue::FromString(read_string(read_unsigned(2)));
      case 0xdb:
        return MsgpackValue::FromString(read_string(read_unsigned(4)));
      case 0xdc:
        return parse_array(read_unsigned(2));
      case 0xdd:
        return parse_array(read_unsigned(4));
      case 0xde:
        return parse_map(read_unsigned(2));
      case 0xdf:
        return parse_map(read_unsigned(4));
      default:
        break;
    }
    throw std::runtime_error("Unsupported msgpack prefix: " +
                             std::to_string(prefix));
  }

  MsgpackValue parse_array(size_t length) {
    std::vector<MsgpackValue> elements;
    elements.reserve(length);
    for (size_t i = 0; i < length; ++i) {
      elements.emplace_back(parse_value());
    }
    return MsgpackValue::FromArray(std::move(elements));
  }

  MsgpackValue parse_map(size_t length) {
    std::vector<std::pair<std::string, MsgpackValue>> entries;
    entries.reserve(length);
    for (size_t i = 0; i < length; ++i) {
      MsgpackValue key = parse_value();
      if (key.kind != MsgpackValue::Kind::String) {
        throw std::runtime_error("Expected string key in msgpack map");
      }
      entries.emplace_back(std::move(key.string_value), parse_value());
    }
    return MsgpackValue::FromMap(std::move(entries));
  }

  uint8_t read_byte() {
    if (offset_ >= size_) {
      throw std::runtime_error("Unexpected end of msgpack payload");
    }
    return data_[offset_++];
  }

  uint64_t read_unsigned(size_t num_bytes) {
    if (num_bytes == 0 || num_bytes > sizeof(uint64_t)) {
      throw std::runtime_error("Invalid msgpack integer length");
    }
    if (offset_ + num_bytes > size_) {
      throw std::runtime_error("Unexpected end of msgpack payload");
    }
    uint64_t value = 0;
    for (size_t i = 0; i < num_bytes; ++i) {
      value = (value << 8) | data_[offset_ + i];
    }
    offset_ += num_bytes;
    return value;
  }

  static int64_t sign_extend(uint64_t value, size_t bits) {
    const uint64_t mask = uint64_t{1} << (bits - 1);
    if (value & mask) {
      const uint64_t extend = (~uint64_t{0}) << bits;
      return static_cast<int64_t>(value | extend);
    }
    return static_cast<int64_t>(value);
  }

  int64_t read_signed(size_t num_bytes) {
    uint64_t raw = read_unsigned(num_bytes);
    size_t bits = num_bytes * 8;
    return sign_extend(raw, bits);
  }

  std::string read_string(size_t length) {
    if (offset_ + length > size_) {
      throw std::runtime_error("Unexpected end of msgpack payload");
    }
    std::string out(reinterpret_cast<const char*>(data_ + offset_), length);
    offset_ += length;
    return out;
  }

  double read_float32() {
    uint32_t bits = static_cast<uint32_t>(read_unsigned(4));
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return static_cast<double>(value);
  }

  double read_float64() {
    uint64_t bits = read_unsigned(8);
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

  const uint8_t* data_;
  size_t size_;
  size_t offset_ = 0;
};

struct KVCacheSpecNative {
  int64_t block_size = 0;
};

struct KVCacheGroupSpecNative {
  std::vector<std::string> layer_names;
  KVCacheSpecNative kv_cache_spec;
};

struct SchedulerConfigSnapshotNative {
  int64_t max_num_batched_tokens = 0;
  int64_t max_num_seqs = 0;
  int64_t max_model_len = 0;
  int64_t long_prefill_token_threshold = 0;
  bool chunked_prefill_enabled = false;
  std::string policy;
};

struct SchedulerKVCacheSnapshotNative {
  int64_t num_gpu_blocks = 0;
  int64_t block_size = 0;
  std::vector<KVCacheGroupSpecNative> kv_cache_groups;
  double kv_cache_usage = 0.0;
  int64_t kv_cache_total_blocks = 0;
  int64_t kv_cache_free_blocks = 0;
};

struct SchedulerParallelSnapshotNative {
  int64_t decode_context_parallel_size = 1;
};

struct RequestStateSnapshotNative {
  std::string request_id;
  std::string status;
  int64_t priority = 0;
  double arrival_time = 0.0;
  int64_t num_prompt_tokens = 0;
  int64_t num_computed_tokens = 0;
  int64_t num_output_target_tokens = 0;
  int64_t num_prompt_processed_tokens = 0;
  int64_t num_output_processed_tokens = 0;
  int64_t max_tokens = 0;
  int64_t num_preemptions = 0;
  int64_t num_cached_tokens = 0;
  bool is_long_prompt = false;
  std::vector<int64_t> kv_block_counts;
};

struct SchedulerStateSnapshotNative {
  int64_t version = 0;
  double created_at = 0.0;
  int64_t num_running = 0;
  int64_t num_waiting = 0;
  std::vector<std::string> running_request_ids;
  std::vector<std::string> waiting_request_ids;
  std::unordered_map<std::string, RequestStateSnapshotNative> requests;
  SchedulerConfigSnapshotNative config;
  SchedulerKVCacheSnapshotNative kv_cache_config;
  SchedulerParallelSnapshotNative parallel_config;
  int64_t resident_set_size = 0;
  int64_t waiting_set_size = 0;
  int64_t prefill_backlog_running_tokens = 0;
  int64_t prefill_backlog_running_sq_sum_tokens = 0;
  int64_t prefill_backlog_waiting_tokens = 0;
  int64_t prefill_backlog_waiting_sq_sum_tokens = 0;
  int64_t prefill_backlog_total_tokens = 0;
  int64_t prefill_backlog_total_sq_sum_tokens = 0;
  int64_t decode_backlog_total_tokens = 0;
  int64_t running_context_length_sum_snapshot = 0;
  int64_t running_context_length_sq_sum_snapshot = 0;
  double build_latency_ms = 0.0;
};

class SimRequestState {
 public:
  explicit SimRequestState(const RequestStateSnapshotNative& snapshot)
      : snapshot_(snapshot),
        status_(NormalizeStatus(snapshot.status)),
        num_computed_tokens_(snapshot.num_computed_tokens),
        prefill_processed_(snapshot.num_prompt_processed_tokens),
        decode_processed_(snapshot.num_output_processed_tokens),
        num_preemptions_(snapshot.num_preemptions),
        num_cached_tokens_(snapshot.num_cached_tokens) {}

  const std::string& request_id() const { return snapshot_.request_id; }
  const std::string& status() const { return status_; }
  void set_status(const std::string& status) { status_ = status; }

  int64_t priority() const { return snapshot_.priority; }
  double arrival_time() const { return snapshot_.arrival_time; }

  int64_t num_tokens() const {
    return snapshot_.num_prompt_tokens + decode_processed_;
  }

  int64_t num_computed_tokens() const { return num_computed_tokens_; }
  void set_num_computed_tokens(int64_t value) { num_computed_tokens_ = value; }

  int64_t prefill_remaining() const {
    return std::max<int64_t>(
        0, snapshot_.num_prompt_tokens - prefill_processed_);
  }

  int64_t decode_remaining() const {
    return std::max<int64_t>(
        0, snapshot_.num_output_target_tokens - decode_processed_);
  }

  int64_t prefill_processed() const { return prefill_processed_; }
  int64_t decode_processed() const { return decode_processed_; }
  int64_t num_cached_tokens() const { return num_cached_tokens_; }
  void set_num_cached_tokens(int64_t value) { num_cached_tokens_ = value; }
  int64_t num_preemptions() const { return num_preemptions_; }
  void increment_preemptions() { ++num_preemptions_; }

  void ensure_decode_backlog() {
    if (prefill_remaining() == 0 && decode_remaining() > 0) {
      int64_t target_prefix =
          snapshot_.num_prompt_tokens + decode_processed_;
      if (num_computed_tokens_ >= target_prefix) {
        num_computed_tokens_ = target_prefix - 1;
      }
    }
  }

  std::pair<int64_t, int64_t> consume(int64_t num_tokens) {
    int64_t prefill_consumed =
        std::min<int64_t>(num_tokens, prefill_remaining());
    if (prefill_consumed > 0) {
      prefill_processed_ += prefill_consumed;
      num_tokens -= prefill_consumed;
    }
    int64_t decode_consumed =
        std::min<int64_t>(num_tokens, decode_remaining());
    if (decode_consumed > 0) {
      num_tokens -= decode_consumed;
    }
    int64_t consumed = prefill_consumed + decode_consumed;
    if (consumed > 0) {
      num_computed_tokens_ += consumed;
    }
    return {prefill_consumed, decode_consumed};
  }

  void mark_scheduled(double current_time_ms) {
    if (!first_scheduled_time_ms_) {
      first_scheduled_time_ms_ = current_time_ms;
    }
  }

  void mark_prefill_done(double completed_time_ms) {
    if (!prefill_done_time_ms_) {
      prefill_done_time_ms_ = completed_time_ms;
    }
  }

  void mark_finished(double completed_time_ms) {
    if (!finished_time_ms_) {
      finished_time_ms_ = completed_time_ms;
      status_ = "FINISHED";
    }
  }

  bool is_finished() const {
    return prefill_remaining() == 0 && decode_remaining() == 0;
  }

  void set_prefill_processed(int64_t value) { prefill_processed_ = value; }
  void set_decode_processed(int64_t value) { decode_processed_ = value; }
  void add_decode_processed(int64_t delta) { decode_processed_ += delta; }

 private:
  static std::string NormalizeStatus(const std::string& status) {
    if (status.find("WAITING") != std::string::npos) {
      return "WAITING";
    }
    if (status.find("FINISHED") != std::string::npos) {
      return "FINISHED";
    }
    return status;
  }

  RequestStateSnapshotNative snapshot_;
  std::string status_;
  int64_t num_computed_tokens_;
  int64_t prefill_processed_;
  int64_t decode_processed_;
  int64_t num_preemptions_;
  int64_t num_cached_tokens_;
  std::optional<double> first_scheduled_time_ms_;
  std::optional<double> prefill_done_time_ms_;
  std::optional<double> finished_time_ms_;
};

class SimulationContext {
 public:
  explicit SimulationContext(
      const SchedulerStateSnapshotNative& snapshot,
      std::optional<int64_t> dummy_prompt_tokens_override = std::nullopt)
      : snapshot_(snapshot),
        kv_free_blocks_(snapshot.kv_cache_config.kv_cache_free_blocks),
        current_time_ms_(0.0),
        total_prefill_tokens_(0),
        total_decode_tokens_(0),
        num_batches_(0) {
    requests_.reserve(snapshot.requests.size());
    for (const auto& entry : snapshot.requests) {
      RequestStateSnapshotNative request_snapshot = entry.second;
      if (entry.first == "__DUMMY__" && dummy_prompt_tokens_override.has_value()) {
        request_snapshot.num_prompt_tokens =
            std::max<int64_t>(0, *dummy_prompt_tokens_override);
      }
      requests_.emplace(entry.first, SimRequestState(request_snapshot));
      kv_allocations_[entry.first] =
          entry.second.kv_block_counts.empty()
              ? std::vector<int64_t>(num_kv_groups(), 0)
              : entry.second.kv_block_counts;
    }
    for (const auto& req_id : snapshot.running_request_ids) {
      auto* req = find_request(req_id);
      if (req) {
        running_.push_back(req);
      }
    }
    for (const auto& req_id : snapshot.waiting_request_ids) {
      auto* req = find_request(req_id);
      if (req) {
        waiting_.push_back(req);
      }
    }
    running_at_snapshot_ = snapshot.running_request_ids.size();
    queued_at_snapshot_ =
        snapshot.waiting_request_ids.size() > 0
            ? static_cast<int64_t>(snapshot.waiting_request_ids.size() - 1)
            : 0;
  }

  const SchedulerStateSnapshotNative& snapshot() const { return snapshot_; }

  std::vector<SimRequestState*>& running() { return running_; }
  const std::vector<SimRequestState*>& running() const { return running_; }

  std::deque<SimRequestState*>& waiting() { return waiting_; }
  const std::deque<SimRequestState*>& waiting() const { return waiting_; }

  double& current_time_ms() { return current_time_ms_; }
  double current_time_ms() const { return current_time_ms_; }
  int64_t& total_prefill_tokens() { return total_prefill_tokens_; }
  int64_t total_prefill_tokens_value() const { return total_prefill_tokens_; }
  int64_t& total_decode_tokens() { return total_decode_tokens_; }
  int64_t total_decode_tokens_value() const { return total_decode_tokens_; }
  int64_t& num_batches() { return num_batches_; }
  int64_t num_batches_value() const { return num_batches_; }

  int64_t running_at_snapshot() const { return running_at_snapshot_; }
  int64_t queued_at_snapshot() const { return queued_at_snapshot_; }

  SimRequestState* get_request(const std::string& request_id) {
    return find_request(request_id);
  }

  int64_t num_kv_groups() const {
    return static_cast<int64_t>(snapshot_.kv_cache_config.kv_cache_groups.size());
  }

  int64_t block_size() const { return snapshot_.kv_cache_config.block_size; }

  const std::vector<int64_t>& allocated_blocks(
      const SimRequestState& req) const {
    auto it = kv_allocations_.find(req.request_id());
    if (it == kv_allocations_.end()) {
      static const std::vector<int64_t> kEmpty;
      return kEmpty;
    }
    return it->second;
  }

  void free_request_blocks(SimRequestState& req) {
    auto& current = kv_allocations_[req.request_id()];
    int64_t reclaimed = 0;
    for (int64_t blocks : current) {
      reclaimed += blocks;
    }
    kv_free_blocks_ += reclaimed;
    current.assign(num_kv_groups(), 0);
  }

  std::vector<std::vector<int64_t>> create_empty_block_list() const {
    return std::vector<std::vector<int64_t>>(num_kv_groups());
  }

  std::pair<std::vector<std::vector<int64_t>>, int64_t> get_computed_blocks(
      SimRequestState& /*req*/) {
    return {create_empty_block_list(), 0};
  }

  std::vector<int64_t>& get_blocks(const std::string& request_id) {
    auto it = kv_allocations_.find(request_id);
    if (it == kv_allocations_.end()) {
      auto inserted = kv_allocations_.emplace(
          request_id, std::vector<int64_t>(num_kv_groups(), 0));
      return inserted.first->second;
    }
    return it->second;
  }

  std::optional<std::vector<int64_t>> allocate_slots(
      SimRequestState& req,
      int64_t num_new_tokens,
      int64_t /*num_new_computed_tokens*/ = 0,
      const std::vector<std::vector<int64_t>>* /*new_computed_blocks*/ = nullptr,
      int64_t /*num_lookahead_tokens*/ = 0,
      bool /*delay_cache_blocks*/ = false,
      int64_t /*num_encoder_tokens*/ = 0) {
    int64_t block_size = this->block_size();
    if (block_size <= 0) {
      return std::nullopt;
    }
    auto& current_blocks = get_blocks(req.request_id());
    int64_t total_tokens_after = req.num_computed_tokens() + num_new_tokens;
    int64_t group_count = num_kv_groups();
    size_t required_size = static_cast<size_t>(group_count);
    ensure_block_scratch(required_size);
    auto& required = block_requirement_scratch_;
    auto& additional = block_additional_scratch_;
    std::fill_n(required.begin(), required_size, int64_t{0});
    std::fill_n(additional.begin(), required_size, int64_t{0});
    for (int64_t i = 0; i < group_count; ++i) {
      size_t idx = static_cast<size_t>(i);
      required[idx] =
          (total_tokens_after + block_size - 1) / block_size;
    }
    int64_t additional_total = 0;
    for (int64_t i = 0; i < group_count; ++i) {
      size_t idx = static_cast<size_t>(i);
      additional[idx] =
          std::max<int64_t>(0, required[idx] - current_blocks[idx]);
      additional_total += additional[idx];
    }
    if (additional_total > kv_free_blocks_) {
      return std::nullopt;
    }
    kv_free_blocks_ -= additional_total;
    for (int64_t i = 0; i < group_count; ++i) {
      size_t idx = static_cast<size_t>(i);
      current_blocks[idx] += additional[idx];
    }
    return current_blocks;
  }

  bool can_allocate_slots(SimRequestState& req, int64_t num_new_tokens) {
    int64_t block_size = this->block_size();
    if (block_size <= 0) {
      return false;
    }
    const auto& current_blocks = get_blocks(req.request_id());
    int64_t total_tokens_after = req.num_computed_tokens() + num_new_tokens;
    int64_t group_count = num_kv_groups();
    size_t required_size = static_cast<size_t>(group_count);
    ensure_block_scratch(required_size);
    auto& required = block_requirement_scratch_;
    std::fill_n(required.begin(), required_size, int64_t{0});
    int64_t additional_total = 0;
    for (int64_t i = 0; i < group_count; ++i) {
      size_t idx = static_cast<size_t>(i);
      required[idx] =
          (total_tokens_after + block_size - 1) / block_size;
      additional_total +=
          std::max<int64_t>(0, required[idx] - current_blocks[idx]);
    }
    if (additional_total > kv_free_blocks_) {
      return false;
    }
    return true;
  }

 private:
  SimRequestState* find_request(const std::string& request_id) {
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
      return nullptr;
    }
    return &it->second;
  }

  const SchedulerStateSnapshotNative snapshot_;
  std::unordered_map<std::string, SimRequestState> requests_;
  std::vector<SimRequestState*> running_;
  std::deque<SimRequestState*> waiting_;
  std::unordered_map<std::string, std::vector<int64_t>> kv_allocations_;
  int64_t kv_free_blocks_;
  double current_time_ms_;
  int64_t total_prefill_tokens_;
  int64_t total_decode_tokens_;
  int64_t num_batches_;
  int64_t queued_at_snapshot_ = 0;
  int64_t running_at_snapshot_ = 0;
  std::vector<int64_t> block_requirement_scratch_;
  std::vector<int64_t> block_additional_scratch_;

  void ensure_block_scratch(size_t count) {
    if (block_requirement_scratch_.size() < count) {
      block_requirement_scratch_.resize(count);
    }
    if (block_additional_scratch_.size() < count) {
      block_additional_scratch_.resize(count);
    }
  }
};

struct SimulationMetadataNative {
  int64_t num_batches = 0;
  int64_t num_running = 0;
  int64_t num_waiting = 0;
  int64_t running_at_snapshot = 0;
  int64_t queued_at_snapshot = 0;
  int64_t total_prefill_tokens = 0;
  int64_t total_decode_tokens = 0;
  double estimated_wait_ms = 0.0;
};

struct SimulationOutcome {
  int64_t snapshot_version = 0;
  double snapshot_timestamp = 0.0;
  double simulation_timestamp = 0.0;
  int64_t num_requests = 0;
  double snapshot_build_latency_ms = 0.0;
  double simulation_latency_ms = 0.0;
  SimulationMetadataNative metadata;
};

static py::dict MetadataToPyDict(
    const SimulationMetadataNative& metadata) {
  py::dict dict;
  dict["num_batches"] = metadata.num_batches;
  dict["num_running"] = metadata.num_running;
  dict["num_waiting"] = metadata.num_waiting;
  dict["running_at_snapshot"] = metadata.running_at_snapshot;
  dict["queued_at_snapshot"] = metadata.queued_at_snapshot;
  dict["total_prefill_tokens"] = metadata.total_prefill_tokens;
  dict["total_decode_tokens"] = metadata.total_decode_tokens;
  dict["estimated_wait_ms"] = metadata.estimated_wait_ms;
  return dict;
}

template <typename Container>
void RemoveAll(Container& container,
               const std::vector<SimRequestState*>& targets) {
  if (targets.empty()) {
    return;
  }
  thread_local std::vector<SimRequestState*> removal_scratch;
  removal_scratch.assign(targets.begin(), targets.end());
  std::sort(removal_scratch.begin(), removal_scratch.end());

  container.erase(
      std::remove_if(container.begin(), container.end(),
                     [&](SimRequestState* item) {
                       return std::binary_search(removal_scratch.begin(),
                                                 removal_scratch.end(), item);
                     }),
      container.end());
}

struct BatchBuildOutput {
  std::vector<SimRequestState*> batch_requests;
  std::vector<int64_t> batch_tokens;
  int64_t sum_context_length = 0;
  int64_t sum_sq_tokens = 0;
};

static BatchBuildOutput BuildNextBatch(SimulationContext& state) {
  const auto& config = state.snapshot().config;
  BatchBuildOutput output;
  auto& batch_requests = output.batch_requests;
  auto& batch_tokens = output.batch_tokens;

  std::vector<SimRequestState*> scheduled_new_reqs;
  std::vector<SimRequestState*> scheduled_resumed_reqs;
  std::vector<SimRequestState*> scheduled_running_reqs;
  std::vector<SimRequestState*> preempted_reqs;
  std::vector<std::pair<SimRequestState*, std::vector<int64_t>>>
      req_to_new_blocks;
  const size_t prealloc =
      static_cast<size_t>(std::max<int64_t>(int64_t{0}, config.max_num_seqs));
  req_to_new_blocks.reserve(prealloc);
  std::vector<int64_t> scheduled_token_counts;
  scheduled_token_counts.reserve(prealloc);

  int64_t token_budget = config.max_num_batched_tokens;
  size_t req_index = 0;
  auto& running = state.running();

  while (req_index < running.size()) {
    SimRequestState* request = running[req_index];
    request->ensure_decode_backlog();

    if (token_budget <= 0) {
      break;
    }
    if (request->is_finished()) {
      ++req_index;
      continue;
    }

    int64_t num_new_tokens =
        request->num_tokens() - request->num_computed_tokens();
    if (num_new_tokens <= 0) {
      throw std::runtime_error(
          "Request has no new tokens to schedule");
    }

    if (config.long_prefill_token_threshold > 0 &&
        config.long_prefill_token_threshold < num_new_tokens) {
      num_new_tokens = config.long_prefill_token_threshold;
    }

    int64_t remaining_model_budget =
        config.max_model_len - request->num_computed_tokens();
    num_new_tokens = std::min<int64_t>(
        num_new_tokens, std::min(token_budget, remaining_model_budget));

    if (num_new_tokens <= 0) {
      ++req_index;
      continue;
    }

    std::optional<std::vector<int64_t>> new_blocks;
    while (true) {
      new_blocks = state.allocate_slots(*request, num_new_tokens);
      if (new_blocks.has_value()) {
        break;
      }

      SimRequestState* preempted_req = nullptr;
      if (config.policy == "priority") {
        throw std::runtime_error(
            "Priority scheduling not implemented yet");
      } else {
        if (running.empty()) {
          break;
        }
        preempted_req = running.back();
        running.pop_back();
      }

      state.free_request_blocks(*preempted_req);
      preempted_req->set_status("PREEMPTED");
      preempted_req->set_num_computed_tokens(0);
      preempted_req->set_prefill_processed(0);
      preempted_req->increment_preemptions();

      if (config.policy == "priority") {
        throw std::runtime_error(
            "Priority scheduling not implemented yet");
      } else {
        state.waiting().push_front(preempted_req);
      }

      preempted_reqs.push_back(preempted_req);
      if (preempted_req == request) {
        break;
      }
    }

    if (!new_blocks.has_value()) {
      break;
    }

    scheduled_running_reqs.push_back(request);
    req_to_new_blocks.emplace_back(request, std::move(*new_blocks));
    scheduled_token_counts.push_back(num_new_tokens);
    token_budget -= num_new_tokens;
    ++req_index;

    batch_requests.push_back(request);
    batch_tokens.push_back(num_new_tokens);
    int64_t request_tokens = request->num_tokens();
    output.sum_context_length += request_tokens;
    output.sum_sq_tokens += request_tokens * request_tokens;
  }

  std::deque<SimRequestState*> skipped_waiting_requests;

  if (preempted_reqs.empty()) {
    while (!state.waiting().empty()) {
      if (token_budget <= 0) {
        break;
      }
      if (static_cast<int64_t>(state.running().size()) >=
          config.max_num_seqs) {
        break;
      }

      SimRequestState* request = nullptr;
      if (config.policy == "priority") {
        throw std::runtime_error(
            "Priority scheduling not implemented yet");
      } else {
        request = state.waiting().front();
      }
      request->ensure_decode_backlog();

      std::vector<std::vector<int64_t>> new_computed_blocks;
      int64_t num_new_local_computed_tokens = 0;
      int64_t num_computed_tokens = 0;
      if (request->num_computed_tokens() == 0) {
        auto computed = state.get_computed_blocks(*request);
        new_computed_blocks = std::move(computed.first);
        num_new_local_computed_tokens = computed.second;
        num_computed_tokens = num_new_local_computed_tokens;
      } else {
        new_computed_blocks = state.create_empty_block_list();
        num_new_local_computed_tokens = 0;
        num_computed_tokens = request->num_computed_tokens();
      }

      int64_t num_new_tokens =
          request->num_tokens() - num_computed_tokens;
      if (config.long_prefill_token_threshold > 0 &&
          config.long_prefill_token_threshold < num_new_tokens) {
        num_new_tokens = config.long_prefill_token_threshold;
      }

      if (!config.chunked_prefill_enabled &&
          num_new_tokens > token_budget) {
        if (config.policy == "priority") {
          throw std::runtime_error(
              "Priority scheduling not implemented yet");
        } else {
          state.waiting().pop_front();
          skipped_waiting_requests.push_front(request);
        }
        continue;
      }

      num_new_tokens = std::min<int64_t>(num_new_tokens, token_budget);
      if (num_new_tokens <= 0) {
        break;
      }

      auto new_blocks =
          state.allocate_slots(*request, num_new_tokens,
                               num_new_local_computed_tokens,
                               &new_computed_blocks);
      if (!new_blocks.has_value()) {
        break;
      }

      if (config.policy == "priority") {
        throw std::runtime_error(
            "Priority scheduling not implemented yet");
      } else {
        request = state.waiting().front();
        state.waiting().pop_front();
      }

      state.running().push_back(request);

      if (request->status() == "WAITING") {
        scheduled_new_reqs.push_back(request);
      } else if (request->status() == "PREEMPTED") {
        scheduled_resumed_reqs.push_back(request);
      } else {
        throw std::runtime_error("Invalid request status: " +
                                 request->status());
      }

      req_to_new_blocks.emplace_back(request,
                                     state.get_blocks(request->request_id()));
      scheduled_token_counts.push_back(num_new_tokens);
      token_budget -= num_new_tokens;
      request->set_status("RUNNING");
      request->set_num_computed_tokens(num_computed_tokens);
      if (request->num_cached_tokens() < 0) {
        request->set_num_cached_tokens(num_computed_tokens);
      }

      batch_requests.push_back(request);
      batch_tokens.push_back(num_new_tokens);
      int64_t request_tokens = request->num_tokens();
      output.sum_context_length += request_tokens;
      output.sum_sq_tokens += request_tokens * request_tokens;
    }
  }

  if (!skipped_waiting_requests.empty()) {
    if (config.policy == "priority") {
      throw std::runtime_error(
          "Priority scheduling not implemented yet");
    } else {
      for (SimRequestState* req : skipped_waiting_requests) {
        state.waiting().push_front(req);
      }
    }
  }

  int64_t total_num_scheduled_tokens = 0;
  for (int64_t tokens : scheduled_token_counts) {
    total_num_scheduled_tokens += tokens;
  }
  if (total_num_scheduled_tokens > config.max_num_batched_tokens) {
    throw std::runtime_error(
        "Scheduled tokens exceed max_num_batched_tokens");
  }
  if (token_budget < 0) {
    throw std::runtime_error("Token budget went negative");
  }
  size_t scheduled_total =
      scheduled_new_reqs.size() + scheduled_resumed_reqs.size() +
      scheduled_running_reqs.size();
  if (scheduled_total > state.running().size()) {
    throw std::runtime_error(
        "Scheduled requests exceed running queue length");
  }

  return output;
}

struct ApplyBatchOutput {
  int64_t total_prefill = 0;
  int64_t total_prefill_sq_sum = 0;
  int64_t total_decode = 0;
  std::vector<SimRequestState*> prefill_done;
  std::vector<SimRequestState*> finished;
};

static ApplyBatchOutput ApplyBatchResultNative(
    SimulationContext& state,
    const std::vector<SimRequestState*>& batch_requests,
    const std::vector<int64_t>& batch_tokens) {
  ApplyBatchOutput result;
  const auto& config = state.snapshot().config;

  std::vector<SimRequestState*> stopped_running_reqs;
  std::vector<SimRequestState*> stopped_preempted_reqs;

  for (size_t i = 0; i < batch_requests.size(); ++i) {
    SimRequestState* request = batch_requests[i];
    int64_t num_tokens_scheduled = batch_tokens[i];
    if (num_tokens_scheduled <= 0 || request == nullptr) {
      continue;
    }

    int64_t prefill_before = request->prefill_remaining();
    int64_t decode_before = request->decode_remaining();
    auto consumed = request->consume(num_tokens_scheduled);
    int64_t consumed_prefill = consumed.first;
    int64_t consumed_decode = consumed.second;
    result.total_prefill += consumed_prefill;
    result.total_prefill_sq_sum += consumed_prefill * consumed_prefill;
    result.total_decode += consumed_decode;

    int64_t emitted_decode = consumed_decode;
    if (prefill_before > 0 && request->prefill_remaining() == 0 &&
        decode_before > 0) {
      emitted_decode += 1;
    }

    if (emitted_decode > 0) {
      emitted_decode =
          std::min<int64_t>(emitted_decode, request->decode_remaining());
      request->add_decode_processed(emitted_decode);
    }

    if (prefill_before > 0 && request->prefill_remaining() == 0) {
      result.prefill_done.push_back(request);
    }
    if (decode_before > 0 && request->is_finished()) {
      std::string status_before_stop = request->status();
      state.free_request_blocks(*request);
      request->set_status("FINISHED");
      if (status_before_stop == "RUNNING") {
        stopped_running_reqs.push_back(request);
      } else {
        stopped_preempted_reqs.push_back(request);
      }
    }
  }

  if (!stopped_running_reqs.empty()) {
    RemoveAll(state.running(), stopped_running_reqs);
  }
  if (!stopped_preempted_reqs.empty()) {
    RemoveAll(state.waiting(), stopped_preempted_reqs);
  }

  result.finished = stopped_running_reqs;
  result.finished.insert(result.finished.end(),
                         stopped_preempted_reqs.begin(),
                         stopped_preempted_reqs.end());

  (void)config;
  return result;
}

static double EstimateBatchTime(int64_t num_prefill_tokens,
                                int64_t num_prefill_sq_sum,
                                int64_t num_decode_tokens,
                                int64_t sum_context_length,
                                int64_t sum_sq_tokens,
                                double intercept,
                                double prefill_coeff,
                                double prefill_sq_coeff,
                                double decode_coeff,
                                double sum_coeff,
                                double sum_sq_coeff) {
  if (num_prefill_tokens <= 0 && num_decode_tokens <= 0) {
    throw std::runtime_error("Batch must have at least one token");
  }
  return (intercept + (prefill_coeff * num_prefill_tokens) +
          (prefill_sq_coeff * num_prefill_sq_sum) +
          (decode_coeff * num_decode_tokens) +
          (sum_coeff * sum_context_length) + (sum_sq_coeff * sum_sq_tokens)) *
         1000;
}

static SimulationMetadataNative RunSimulationNative(
    const SchedulerStateSnapshotNative& snapshot,
    double intercept,
    double prefill_coeff,
    double prefill_sq_coeff,
    double decode_coeff,
    double sum_coeff,
    double sum_sq_coeff,
    std::optional<int64_t> dummy_prompt_tokens_override = std::nullopt) {
  SimulationContext state(snapshot, dummy_prompt_tokens_override);
  int idle_ticks = 0;
  const int max_idle_ticks = 4;

  while (true) {
    auto batch = BuildNextBatch(state);
    bool contains_dummy = false;
    for (SimRequestState* req : batch.batch_requests) {
      if (req && req->request_id() == "__DUMMY__") {
        contains_dummy = true;
        break;
      }
    }
    if (contains_dummy) {
      break;
    }

    if (batch.batch_requests.empty()) {
      idle_ticks += 1;
      if (idle_ticks >= max_idle_ticks) {
        throw std::runtime_error(
            "Simulation stalled: no requests can be scheduled");
      }
      continue;
    }
    idle_ticks = 0;

    double start_time = state.current_time_ms();
    auto apply_result =
        ApplyBatchResultNative(state, batch.batch_requests, batch.batch_tokens);
    double batch_time = EstimateBatchTime(
        apply_result.total_prefill, apply_result.total_prefill_sq_sum,
        apply_result.total_decode, batch.sum_context_length, batch.sum_sq_tokens,
        intercept, prefill_coeff, prefill_sq_coeff, decode_coeff, sum_coeff,
        sum_sq_coeff);
    double end_time = start_time + batch_time;

    for (SimRequestState* req : apply_result.finished) {
      req->mark_finished(end_time);
    }

    state.total_prefill_tokens() += apply_result.total_prefill;
    state.total_decode_tokens() += apply_result.total_decode;
    state.current_time_ms() = end_time;
    state.num_batches() += 1;
  }

  SimulationMetadataNative metadata;
  metadata.num_batches = state.num_batches_value();
  metadata.num_running =
      static_cast<int64_t>(state.running().size());
  metadata.num_waiting =
      static_cast<int64_t>(state.waiting().size());
  metadata.running_at_snapshot = state.running_at_snapshot();
  metadata.queued_at_snapshot = state.queued_at_snapshot();
  metadata.total_prefill_tokens = state.total_prefill_tokens_value();
  metadata.total_decode_tokens = state.total_decode_tokens_value();
  metadata.estimated_wait_ms = state.current_time_ms();
  return metadata;
}

static const MsgpackValue& RequireKind(const MsgpackValue& value,
                                       MsgpackValue::Kind kind,
                                       const char* context) {
  if (value.kind != kind) {
    throw std::runtime_error(std::string("Expected ") + context);
  }
  return value;
}

static const MsgpackValue& GetMapValue(const MsgpackValue& map,
                                       const std::string& key) {
  RequireKind(map, MsgpackValue::Kind::Map, "map");
  for (const auto& entry : map.map_value) {
    if (entry.first == key) {
      return entry.second;
    }
  }
  throw std::runtime_error("Missing key in snapshot: " + key);
}

static const MsgpackValue* FindMapValue(const MsgpackValue& map,
                                        const std::string& key) {
  RequireKind(map, MsgpackValue::Kind::Map, "map");
  for (const auto& entry : map.map_value) {
    if (entry.first == key) {
      return &entry.second;
    }
  }
  return nullptr;
}

static int64_t ToInt(const MsgpackValue& value, const char* field) {
  switch (value.kind) {
    case MsgpackValue::Kind::Int:
      return value.int_value;
    case MsgpackValue::Kind::UInt:
      return static_cast<int64_t>(value.uint_value);
    case MsgpackValue::Kind::Double:
      return static_cast<int64_t>(value.double_value);
    default:
      throw std::runtime_error(std::string("Expected integer for ") + field);
  }
}

static double ToDouble(const MsgpackValue& value, const char* field) {
  switch (value.kind) {
    case MsgpackValue::Kind::Double:
      return value.double_value;
    case MsgpackValue::Kind::Int:
      return static_cast<double>(value.int_value);
    case MsgpackValue::Kind::UInt:
      return static_cast<double>(value.uint_value);
    default:
      throw std::runtime_error(std::string("Expected float for ") + field);
  }
}

static bool ToBool(const MsgpackValue& value, const char* field) {
  if (value.kind == MsgpackValue::Kind::Bool) {
    return value.bool_value;
  }
  if (value.kind == MsgpackValue::Kind::Int) {
    return value.int_value != 0;
  }
  if (value.kind == MsgpackValue::Kind::UInt) {
    return value.uint_value != 0;
  }
  throw std::runtime_error(std::string("Expected bool for ") + field);
}

static std::string ToString(const MsgpackValue& value, const char* field) {
  if (value.kind != MsgpackValue::Kind::String) {
    throw std::runtime_error(std::string("Expected string for ") + field);
  }
  return value.string_value;
}

static std::vector<std::string> ToStringVector(const MsgpackValue& value,
                                               const char* field) {
  RequireKind(value, MsgpackValue::Kind::Array, field);
  std::vector<std::string> out;
  out.reserve(value.array_value.size());
  for (const auto& element : value.array_value) {
    out.emplace_back(ToString(element, field));
  }
  return out;
}

static std::vector<int64_t> ToIntVector(const MsgpackValue& value,
                                        const char* field) {
  RequireKind(value, MsgpackValue::Kind::Array, field);
  std::vector<int64_t> out;
  out.reserve(value.array_value.size());
  for (const auto& element : value.array_value) {
    out.emplace_back(ToInt(element, field));
  }
  return out;
}

static KVCacheSpecNative ParseKVCacheSpec(const MsgpackValue& value) {
  RequireKind(value, MsgpackValue::Kind::Map, "kv_cache_spec");
  KVCacheSpecNative spec;
  spec.block_size = ToInt(GetMapValue(value, "block_size"), "block_size");
  return spec;
}

static KVCacheGroupSpecNative ParseKVGroup(const MsgpackValue& value) {
  RequireKind(value, MsgpackValue::Kind::Map, "kv_cache_group");
  KVCacheGroupSpecNative group;
  group.layer_names =
      ToStringVector(GetMapValue(value, "layer_names"), "layer_names");
  group.kv_cache_spec =
      ParseKVCacheSpec(GetMapValue(value, "kv_cache_spec"));
  return group;
}

static SchedulerConfigSnapshotNative ParseConfig(const MsgpackValue& value) {
  RequireKind(value, MsgpackValue::Kind::Map, "config");
  SchedulerConfigSnapshotNative config;
  config.max_num_batched_tokens =
      ToInt(GetMapValue(value, "max_num_batched_tokens"),
            "max_num_batched_tokens");
  config.max_num_seqs =
      ToInt(GetMapValue(value, "max_num_seqs"), "max_num_seqs");
  config.max_model_len =
      ToInt(GetMapValue(value, "max_model_len"), "max_model_len");
  config.long_prefill_token_threshold =
      ToInt(GetMapValue(value, "long_prefill_token_threshold"),
            "long_prefill_token_threshold");
  config.chunked_prefill_enabled =
      ToBool(GetMapValue(value, "chunked_prefill_enabled"),
             "chunked_prefill_enabled");
  config.policy = ToString(GetMapValue(value, "policy"), "policy");
  return config;
}

static SchedulerKVCacheSnapshotNative ParseKVConfig(const MsgpackValue& value) {
  RequireKind(value, MsgpackValue::Kind::Map, "kv_cache_config");
  SchedulerKVCacheSnapshotNative kv;
  kv.num_gpu_blocks =
      ToInt(GetMapValue(value, "num_gpu_blocks"), "num_gpu_blocks");
  kv.block_size = ToInt(GetMapValue(value, "block_size"), "block_size");
  const auto& groups_value = GetMapValue(value, "kv_cache_groups");
  RequireKind(groups_value, MsgpackValue::Kind::Array, "kv_cache_groups");
  for (const auto& entry : groups_value.array_value) {
    kv.kv_cache_groups.emplace_back(ParseKVGroup(entry));
  }
  kv.kv_cache_usage =
      ToDouble(GetMapValue(value, "kv_cache_usage"), "kv_cache_usage");
  kv.kv_cache_total_blocks =
      ToInt(GetMapValue(value, "kv_cache_total_blocks"),
            "kv_cache_total_blocks");
  kv.kv_cache_free_blocks =
      ToInt(GetMapValue(value, "kv_cache_free_blocks"), "kv_cache_free_blocks");
  return kv;
}

static SchedulerParallelSnapshotNative ParseParallelConfig(
    const MsgpackValue& value) {
  RequireKind(value, MsgpackValue::Kind::Map, "parallel_config");
  SchedulerParallelSnapshotNative parallel;
  parallel.decode_context_parallel_size =
      ToInt(GetMapValue(value, "decode_context_parallel_size"),
            "decode_context_parallel_size");
  return parallel;
}

static RequestStateSnapshotNative ParseRequestSnapshot(
    const std::string& req_id, const MsgpackValue& value) {
  RequireKind(value, MsgpackValue::Kind::Map, "request");
  RequestStateSnapshotNative req;
  req.request_id = req_id;
  req.status = ToString(GetMapValue(value, "status"), "status");
  req.priority = ToInt(GetMapValue(value, "priority"), "priority");
  req.arrival_time =
      ToDouble(GetMapValue(value, "arrival_time"), "arrival_time");
  req.num_prompt_tokens =
      ToInt(GetMapValue(value, "num_prompt_tokens"), "num_prompt_tokens");
  req.num_computed_tokens =
      ToInt(GetMapValue(value, "num_computed_tokens"), "num_computed_tokens");
  req.num_output_target_tokens = ToInt(
      GetMapValue(value, "num_output_target_tokens"),
      "num_output_target_tokens");
  req.num_prompt_processed_tokens =
      ToInt(GetMapValue(value, "num_prompt_processed_tokens"),
            "num_prompt_processed_tokens");
  req.num_output_processed_tokens =
      ToInt(GetMapValue(value, "num_output_processed_tokens"),
            "num_output_processed_tokens");
  req.max_tokens = ToInt(GetMapValue(value, "max_tokens"), "max_tokens");
  req.num_preemptions =
      ToInt(GetMapValue(value, "num_preemptions"), "num_preemptions");
  req.num_cached_tokens =
      ToInt(GetMapValue(value, "num_cached_tokens"), "num_cached_tokens");
  req.is_long_prompt =
      ToBool(GetMapValue(value, "is_long_prompt"), "is_long_prompt");
  req.kv_block_counts =
      ToIntVector(GetMapValue(value, "kv_block_counts"), "kv_block_counts");
  return req;
}

static SchedulerStateSnapshotNative ParseSchedulerSnapshot(
    const MsgpackValue& value) {
  RequireKind(value, MsgpackValue::Kind::Map, "scheduler_snapshot");
  SchedulerStateSnapshotNative snapshot;
  snapshot.version = ToInt(GetMapValue(value, "version"), "version");
  snapshot.created_at =
    ToDouble(GetMapValue(value, "created_at"), "created_at");
  snapshot.num_running = ToInt(GetMapValue(value, "num_running"), "num_running");
  snapshot.num_waiting = ToInt(GetMapValue(value, "num_waiting"), "num_waiting");
  snapshot.running_request_ids =
      ToStringVector(GetMapValue(value, "running_request_ids"),
                     "running_request_ids");
  snapshot.waiting_request_ids =
      ToStringVector(GetMapValue(value, "waiting_request_ids"),
                     "waiting_request_ids");

  const auto& reqs = GetMapValue(value, "requests");
  RequireKind(reqs, MsgpackValue::Kind::Map, "requests");
  for (const auto& entry : reqs.map_value) {
    snapshot.requests.emplace(entry.first,
                              ParseRequestSnapshot(entry.first, entry.second));
  }

  snapshot.config = ParseConfig(GetMapValue(value, "config"));
  snapshot.kv_cache_config =
      ParseKVConfig(GetMapValue(value, "kv_cache_config"));
  snapshot.parallel_config =
      ParseParallelConfig(GetMapValue(value, "parallel_config"));
  if (const auto* field = FindMapValue(value, "resident_set_size")) {
    snapshot.resident_set_size = ToInt(*field, "resident_set_size");
  }
  if (const auto* field = FindMapValue(value, "waiting_set_size")) {
    snapshot.waiting_set_size = ToInt(*field, "waiting_set_size");
  }
  if (const auto* field =
          FindMapValue(value, "prefill_backlog_running_tokens")) {
    snapshot.prefill_backlog_running_tokens =
        ToInt(*field, "prefill_backlog_running_tokens");
  }
  if (const auto* field =
          FindMapValue(value, "prefill_backlog_running_sq_sum_tokens")) {
    snapshot.prefill_backlog_running_sq_sum_tokens =
        ToInt(*field, "prefill_backlog_running_sq_sum_tokens");
  }
  if (const auto* field =
          FindMapValue(value, "prefill_backlog_waiting_tokens")) {
    snapshot.prefill_backlog_waiting_tokens =
        ToInt(*field, "prefill_backlog_waiting_tokens");
  }
  if (const auto* field =
          FindMapValue(value, "prefill_backlog_waiting_sq_sum_tokens")) {
    snapshot.prefill_backlog_waiting_sq_sum_tokens =
        ToInt(*field, "prefill_backlog_waiting_sq_sum_tokens");
  }
  if (const auto* field =
          FindMapValue(value, "prefill_backlog_total_tokens")) {
    snapshot.prefill_backlog_total_tokens =
        ToInt(*field, "prefill_backlog_total_tokens");
  }
  if (const auto* field =
          FindMapValue(value, "prefill_backlog_total_sq_sum_tokens")) {
    snapshot.prefill_backlog_total_sq_sum_tokens =
        ToInt(*field, "prefill_backlog_total_sq_sum_tokens");
  }
  if (const auto* field =
          FindMapValue(value, "decode_backlog_total_tokens")) {
    snapshot.decode_backlog_total_tokens =
        ToInt(*field, "decode_backlog_total_tokens");
  }
  if (const auto* field =
          FindMapValue(value, "running_context_length_sum_snapshot")) {
    snapshot.running_context_length_sum_snapshot =
        ToInt(*field, "running_context_length_sum_snapshot");
  }
  if (const auto* field =
          FindMapValue(value, "running_context_length_sq_sum_snapshot")) {
    snapshot.running_context_length_sq_sum_snapshot =
        ToInt(*field, "running_context_length_sq_sum_snapshot");
  }
  snapshot.build_latency_ms =
      ToDouble(GetMapValue(value, "build_latency_ms"), "build_latency_ms");
  return snapshot;
}

static py::list ToPyList(const std::vector<std::string>& values) {
  py::list result;
  for (const auto& value : values) {
    result.append(value);
  }
  return result;
}

static py::list ToPyIntList(const std::vector<int64_t>& values) {
  py::list result;
  for (const auto& value : values) {
    result.append(value);
  }
  return result;
}

static py::dict ToPyDict(const KVCacheGroupSpecNative& group) {
  py::dict dict;
  dict["layer_names"] = ToPyList(group.layer_names);
  py::dict spec;
  spec["block_size"] = group.kv_cache_spec.block_size;
  dict["kv_cache_spec"] = spec;
  return dict;
}

static py::dict ToPyDict(const SchedulerConfigSnapshotNative& config) {
  py::dict dict;
  dict["max_num_batched_tokens"] = config.max_num_batched_tokens;
  dict["max_num_seqs"] = config.max_num_seqs;
  dict["max_model_len"] = config.max_model_len;
  dict["long_prefill_token_threshold"] =
      config.long_prefill_token_threshold;
  dict["chunked_prefill_enabled"] = config.chunked_prefill_enabled;
  dict["policy"] = config.policy;
  return dict;
}

static py::dict ToPyDict(const SchedulerKVCacheSnapshotNative& kv) {
  py::dict dict;
  dict["num_gpu_blocks"] = kv.num_gpu_blocks;
  dict["block_size"] = kv.block_size;
  py::list groups;
  for (const auto& group : kv.kv_cache_groups) {
    groups.append(ToPyDict(group));
  }
  dict["kv_cache_groups"] = groups;
  dict["kv_cache_usage"] = kv.kv_cache_usage;
  dict["kv_cache_total_blocks"] = kv.kv_cache_total_blocks;
  dict["kv_cache_free_blocks"] = kv.kv_cache_free_blocks;
  return dict;
}

static py::dict ToPyDict(const SchedulerParallelSnapshotNative& parallel) {
  py::dict dict;
  dict["decode_context_parallel_size"] =
      parallel.decode_context_parallel_size;
  return dict;
}

static py::dict ToPyDict(const RequestStateSnapshotNative& req) {
  py::dict dict;
  dict["request_id"] = req.request_id;
  dict["status"] = req.status;
  dict["priority"] = req.priority;
  dict["arrival_time"] = req.arrival_time;
  dict["num_prompt_tokens"] = req.num_prompt_tokens;
  dict["num_computed_tokens"] = req.num_computed_tokens;
  dict["num_output_target_tokens"] = req.num_output_target_tokens;
  dict["num_prompt_processed_tokens"] = req.num_prompt_processed_tokens;
  dict["num_output_processed_tokens"] = req.num_output_processed_tokens;
  dict["max_tokens"] = req.max_tokens;
  dict["num_preemptions"] = req.num_preemptions;
  dict["num_cached_tokens"] = req.num_cached_tokens;
  dict["is_long_prompt"] = req.is_long_prompt;
  dict["kv_block_counts"] = ToPyIntList(req.kv_block_counts);
  return dict;
}

static py::dict ToPyDict(const SchedulerStateSnapshotNative& snapshot) {
  py::dict dict;
  dict["version"] = snapshot.version;
  dict["created_at"] = snapshot.created_at;
  dict["num_running"] = snapshot.num_running;
  dict["num_waiting"] = snapshot.num_waiting;
  dict["running_request_ids"] = ToPyList(snapshot.running_request_ids);
  dict["waiting_request_ids"] = ToPyList(snapshot.waiting_request_ids);
  py::dict requests;
  for (const auto& entry : snapshot.requests) {
    requests[py::str(entry.first)] = ToPyDict(entry.second);
  }
  dict["requests"] = requests;
  dict["config"] = ToPyDict(snapshot.config);
  dict["kv_cache_config"] = ToPyDict(snapshot.kv_cache_config);
  dict["parallel_config"] = ToPyDict(snapshot.parallel_config);
  dict["resident_set_size"] = snapshot.resident_set_size;
  dict["waiting_set_size"] = snapshot.waiting_set_size;
  dict["prefill_backlog_running_tokens"] =
      snapshot.prefill_backlog_running_tokens;
  dict["prefill_backlog_running_sq_sum_tokens"] =
      snapshot.prefill_backlog_running_sq_sum_tokens;
  dict["prefill_backlog_waiting_tokens"] =
      snapshot.prefill_backlog_waiting_tokens;
  dict["prefill_backlog_waiting_sq_sum_tokens"] =
      snapshot.prefill_backlog_waiting_sq_sum_tokens;
  dict["prefill_backlog_total_tokens"] = snapshot.prefill_backlog_total_tokens;
  dict["prefill_backlog_total_sq_sum_tokens"] =
      snapshot.prefill_backlog_total_sq_sum_tokens;
  dict["decode_backlog_total_tokens"] = snapshot.decode_backlog_total_tokens;
  dict["running_context_length_sum_snapshot"] =
      snapshot.running_context_length_sum_snapshot;
  dict["running_context_length_sq_sum_snapshot"] =
      snapshot.running_context_length_sq_sum_snapshot;
  dict["build_latency_ms"] = snapshot.build_latency_ms;
  return dict;
}

}  // namespace

// Placeholder worker that mirrors the Python surface without any logic.
class SchedulerSimulationWorkerStub {
 public:
  SchedulerSimulationWorkerStub(double interval_s,
                                double intercept,
                                double prefill_coeff,
                                double decode_coeff,
                                double sum_coeff,
                                double prefill_sq_coeff = 0.0,
                                double sum_sq_coeff = 0.0)
      : interval_s_(interval_s),
        intercept_(intercept),
        prefill_coeff_(prefill_coeff),
        decode_coeff_(decode_coeff),
        sum_coeff_(sum_coeff),
        prefill_sq_coeff_(prefill_sq_coeff),
        sum_sq_coeff_(sum_sq_coeff) {}

  ~SchedulerSimulationWorkerStub() { StopThread(); }

  void start() { StartThread(); }
  void stop() { StopThread(); }

  void update_snapshot(py::bytes snapshot_bytes) {
    latest_snapshot_bytes_ = snapshot_bytes;
    MsgpackParser parser(latest_snapshot_bytes_.data(),
                         latest_snapshot_bytes_.size());
    auto parsed_snapshot =
        std::make_shared<SchedulerStateSnapshotNative>(
            ParseSchedulerSnapshot(parser.parse()));
    has_snapshot_ = true;

    PendingInfo info;
    info.version = parsed_snapshot->version;
    info.snapshot_timestamp = parsed_snapshot->created_at;
    info.num_requests =
        static_cast<int64_t>(parsed_snapshot->requests.size());
    info.build_latency_ms = parsed_snapshot->build_latency_ms;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      parsed_snapshot_ = parsed_snapshot;
      pending_snapshot_ = parsed_snapshot;
      pending_info_ = info;
    }
    cv_.notify_all();
  }

  py::object latest_result() const { return latest_result_summary(); }

  py::object latest_snapshot() const {
    if (!has_snapshot_) {
      return py::none();
    }
    return py::bytes(latest_snapshot_bytes_);
  }

  py::object parsed_snapshot() const {
    if (!parsed_snapshot_) {
      return py::none();
    }
    return ToPyDict(*parsed_snapshot_);
  }

  py::dict run_simulation_for_test(py::bytes snapshot_bytes) const {
    std::string buffer = snapshot_bytes;
    MsgpackParser parser(buffer.data(), buffer.size());
    auto snapshot = ParseSchedulerSnapshot(parser.parse());
    try {
      auto metadata = RunSimulationNative(snapshot, intercept_,
                                          prefill_coeff_, prefill_sq_coeff_,
                                          decode_coeff_, sum_coeff_,
                                          sum_sq_coeff_);
      return MetadataToPyDict(metadata);
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("Scheduler simulation failed: ") +
                               e.what());
    } catch (...) {
      throw std::runtime_error(
          "Scheduler simulation failed with an unknown error");
    }
  }

  py::object run_simulation_on_latest_snapshot(int64_t prompt_tokens) const {
    if (prompt_tokens < 0) {
      throw std::runtime_error("prompt_tokens must be >= 0");
    }

    std::shared_ptr<const SchedulerStateSnapshotNative> snapshot;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      snapshot = parsed_snapshot_;
    }
    if (!snapshot) {
      return py::none();
    }

    auto run_start = std::chrono::steady_clock::now();
    double simulation_timestamp =
        std::chrono::duration<double>(run_start.time_since_epoch()).count();

    SimulationMetadataNative metadata;
    try {
      metadata =
          RunSimulationNative(*snapshot, intercept_, prefill_coeff_,
                              prefill_sq_coeff_, decode_coeff_, sum_coeff_,
                              sum_sq_coeff_, prompt_tokens);
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("Scheduler simulation failed: ") +
                               e.what());
    } catch (...) {
      throw std::runtime_error(
          "Scheduler simulation failed with an unknown error");
    }

    auto run_end = std::chrono::steady_clock::now();
    double sim_latency_ms =
        std::chrono::duration<double, std::milli>(run_end - run_start).count();

    py::dict metadata_dict = MetadataToPyDict(metadata);
    return py::make_tuple(
        snapshot->version, snapshot->created_at, simulation_timestamp,
        static_cast<int64_t>(snapshot->requests.size()),
        snapshot->build_latency_ms, sim_latency_ms, metadata_dict);
  }

  void clear_latest_snapshot() {
    latest_snapshot_bytes_.clear();
    has_snapshot_ = false;
    parsed_snapshot_.reset();
    std::lock_guard<std::mutex> lock(mutex_);
    pending_snapshot_.reset();
    pending_info_.reset();
  }
  void clear_latest_result() {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_outcome_.reset();
  }
  py::object latest_result_summary() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!latest_outcome_) {
      return py::none();
    }
    const auto& outcome = *latest_outcome_;
    py::dict metadata = MetadataToPyDict(outcome.metadata);
    return py::make_tuple(outcome.snapshot_version, outcome.snapshot_timestamp,
                          outcome.simulation_timestamp, outcome.num_requests,
                          outcome.snapshot_build_latency_ms,
                          outcome.simulation_latency_ms, metadata);
  }

 private:
  struct PendingInfo {
    int64_t version = 0;
    double snapshot_timestamp = 0.0;
    int64_t num_requests = 0;
    double build_latency_ms = 0.0;
  };

  void StartThread() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (worker_started_) {
      return;
    }
    stop_flag_ = false;
    worker_thread_ =
        std::thread(&SchedulerSimulationWorkerStub::RunLoop, this);
    worker_started_ = true;
  }

  void StopThread() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!worker_started_) {
        return;
      }
      stop_flag_ = true;
    }
    cv_.notify_all();
    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }
    worker_started_ = false;
  }

  void RunLoop() {
    while (true) {
      std::shared_ptr<const SchedulerStateSnapshotNative> snapshot;
      PendingInfo info;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] {
          return stop_flag_ || static_cast<bool>(pending_snapshot_);
        });
        if (stop_flag_) {
          break;
        }
        snapshot = pending_snapshot_;
        pending_snapshot_.reset();
        info = pending_info_.value();
        pending_info_.reset();
      }

      auto run_start = std::chrono::steady_clock::now();
      double simulation_timestamp =
          std::chrono::duration<double>(
              run_start.time_since_epoch())
              .count();

      SimulationMetadataNative metadata;
      bool simulation_ok = true;
      try {
        metadata = RunSimulationNative(*snapshot, intercept_,
                                       prefill_coeff_, prefill_sq_coeff_,
                                       decode_coeff_, sum_coeff_,
                                       sum_sq_coeff_);
      } catch (const std::exception& e) {
        simulation_ok = false;
        std::fprintf(stderr,
                     "[scheduler_sim] Simulation failed for snapshot v%lld: %s\n",
                     static_cast<long long>(info.version), e.what());
        std::fflush(stderr);
      } catch (...) {
        simulation_ok = false;
        std::fprintf(stderr,
                     "[scheduler_sim] Simulation failed for snapshot v%lld with an unknown error\n",
                     static_cast<long long>(info.version));
        std::fflush(stderr);
      }
      if (!simulation_ok) {
        continue;
      }
      auto run_end = std::chrono::steady_clock::now();
      double sim_latency_ms =
          std::chrono::duration<double, std::milli>(run_end - run_start)
              .count();

      SimulationOutcome outcome;
      outcome.snapshot_version = info.version;
      outcome.snapshot_timestamp = info.snapshot_timestamp;
      outcome.simulation_timestamp = simulation_timestamp;
      outcome.num_requests = info.num_requests;
      outcome.snapshot_build_latency_ms = info.build_latency_ms;
      outcome.simulation_latency_ms = sim_latency_ms;
      outcome.metadata = metadata;

      {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_outcome_ = outcome;
      }
    }
  }

  double interval_s_;
  double intercept_;
  double prefill_coeff_;
  double decode_coeff_;
  double sum_coeff_;
  double prefill_sq_coeff_;
  double sum_sq_coeff_;
  bool has_snapshot_ = false;
  std::string latest_snapshot_bytes_;
  std::shared_ptr<const SchedulerStateSnapshotNative> parsed_snapshot_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_flag_ = false;
  bool worker_started_ = false;
  std::thread worker_thread_;
  std::shared_ptr<const SchedulerStateSnapshotNative> pending_snapshot_;
  std::optional<PendingInfo> pending_info_;
  std::optional<SimulationOutcome> latest_outcome_;
};

}  // namespace vllm::scheduler_sim

PYBIND11_MODULE(_scheduler_sim, m) {
  py::class_<vllm::scheduler_sim::SchedulerSimulationWorkerStub>(
      m, "SchedulerSimulationWorker")
      .def(py::init<double, double, double, double, double, double, double>(),
           py::arg("interval_s"),
           py::arg("intercept"),
           py::arg("prefill_coeff"),
           py::arg("decode_coeff"),
           py::arg("sum_coeff"),
           py::arg("prefill_sq_coeff") = 0.0,
           py::arg("sum_sq_coeff") = 0.0)
      .def("start", &vllm::scheduler_sim::SchedulerSimulationWorkerStub::start)
      .def("stop", &vllm::scheduler_sim::SchedulerSimulationWorkerStub::stop)
      .def("update_snapshot",
           &vllm::scheduler_sim::SchedulerSimulationWorkerStub::update_snapshot,
           py::arg("snapshot_bytes"))
      .def("latest_result",
           &vllm::scheduler_sim::SchedulerSimulationWorkerStub::latest_result)
      .def("latest_snapshot",
           &vllm::scheduler_sim::SchedulerSimulationWorkerStub::latest_snapshot)
      .def("parsed_snapshot",
           &vllm::scheduler_sim::SchedulerSimulationWorkerStub::parsed_snapshot)
      .def("run_simulation_for_test",
           &vllm::scheduler_sim::SchedulerSimulationWorkerStub::
               run_simulation_for_test,
           py::arg("snapshot_bytes"))
      .def("run_simulation_on_latest_snapshot",
           &vllm::scheduler_sim::SchedulerSimulationWorkerStub::
               run_simulation_on_latest_snapshot,
           py::arg("prompt_tokens"))
      .def("latest_result_summary",
           &vllm::scheduler_sim::SchedulerSimulationWorkerStub::
               latest_result_summary)
      .def("clear_latest_snapshot",
           &vllm::scheduler_sim::SchedulerSimulationWorkerStub::clear_latest_snapshot)
      .def("clear_latest_result",
           &vllm::scheduler_sim::SchedulerSimulationWorkerStub::clear_latest_result);
}
