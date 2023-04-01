#pragma once

#include <glog/logging.h>
#include <Eigen/StdVector>
#include <memory>
#include <unordered_map>

#ifdef NUM_INTRINSICS_PARAMS
constexpr size_t kNumIntrinsicParams = NUM_INTRINSICS_PARAMS;
#else
constexpr size_t kNumIntrinsicParams = 1;
#endif
#ifdef NUM_DISTORTION_PARAMS
constexpr size_t kNumDistortionParams = NUM_DISTORTION_PARAMS;
#else
constexpr size_t kNumDistortionParams = 2;
#endif

constexpr size_t tagoffset_start = 0;
constexpr size_t tagoffset_counter = 5000;
constexpr size_t tagoffset_state_to_master = 10000;
constexpr size_t tagoffset_average_to_worker = 15000;
constexpr size_t tagoffset_worker_to_worker = 20000;

static constexpr uint64_t key_multiplier = 5000;

namespace dba {

struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

inline auto computeDirectedKeyFromIds(const uint64_t id_a, const uint64_t id_b)
    -> uint64_t {
  CHECK_LT(id_a + 1, std::numeric_limits<uint64_t>::max() / key_multiplier);
  return (id_a + 1) * key_multiplier + id_b;
}

inline auto computeKeyFromIds(const uint64_t id_a, const uint64_t id_b)
    -> uint64_t {
  const auto id_lo = (id_a <= id_b) ? id_a + 1 : id_b + 1;
  const auto id_hi = (id_a > id_b) ? id_a + 1 : id_b + 1;
  CHECK_LT(id_lo, std::numeric_limits<uint64_t>::max() / key_multiplier);
  return id_lo * key_multiplier + id_hi;
}

inline auto computeIdsFromKey(const uint64_t key)
    -> std::pair<uint64_t, uint64_t> {
  const auto id_lo = (key / key_multiplier) - 1;
  const auto id_hi = (key % key_multiplier) - 1;
  return {id_lo, id_hi};
}

struct Observation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector2d obs;
  uint64_t mp_id;
  uint64_t frame_id;
};

using ObservationVector =
    std::vector<Observation, Eigen::aligned_allocator<Observation>>;
using ObservationMap = std::unordered_map<
    uint64_t, Observation, std::hash<uint64_t>, std::equal_to<uint64_t>,
    Eigen::aligned_allocator<std::pair<uint64_t, Observation>>>;

template <size_t NI, size_t ND>
struct PoseDual {
  PoseDual() : id_(std::numeric_limits<uint64_t>::max()) { fill(0.0); }
  PoseDual(const uint64_t id) : id_(id) { fill(0.0); }
  auto operator[](const size_t i) -> double& { return vars_[i]; }
  auto operator[](const size_t i) const -> const double& { return vars_[i]; }
  auto setId(const uint64_t id) -> void { id_ = id; }
  auto getId() const -> uint64_t { return id_; }
  auto getPosition() -> double* { return vars_; }
  auto getRotation() -> double* { return vars_ + kNumPosParams; }
  auto getIntrisincs() -> double* {
    return vars_ + kNumPosParams + kNumRotParams;
  }
  auto getDistortion() -> double* {
    return vars_ + kNumPosParams + kNumRotParams + NI;
  }
  auto fill(const double val) -> void {
    for (auto& v : vars_) v = val;
  }
  static constexpr auto getSize() -> size_t {
    return kNumPosParams + kNumRotParams + NI + ND;
  }

 private:
  static constexpr size_t kNumPosParams = 3;
  static constexpr size_t kNumRotParams = 3;

 private:
  uint64_t id_;
  double vars_[kNumPosParams + kNumRotParams + NI + ND];
};

struct MapPointDual {
  MapPointDual() : id_(std::numeric_limits<uint64_t>::max()) {}
  MapPointDual(const uint64_t id) : id_(id) {}
  auto operator[](const size_t i) -> double& { return vars_[i]; }
  auto operator[](const size_t i) const -> const double& { return vars_[i]; }
  auto setId(const uint64_t id) -> void { id_ = id; }
  auto getId() const -> uint64_t { return id_; }
  auto getPosition() -> double* { return vars_; }
  auto fill(const double val) -> void {
    for (auto& v : vars_) v = val;
  }
  static constexpr auto getSize() -> size_t { return kNumParams; }

 private:
  static constexpr size_t kNumParams = 3;

 private:
  uint64_t id_;
  double vars_[kNumParams];
};

using VectorOfVector3 =
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;

template <int N>
using VectorOfVectorN =
    std::vector<Eigen::Matrix<double, N, 1>,
                Eigen::aligned_allocator<Eigen::Matrix<double, N, 1>>>;

// Stolen from ASLAM in order to be able to use uniqe
namespace internal {
template <typename Type>
struct aligned_delete {
  constexpr aligned_delete() noexcept = default;

  template <typename TypeUp,
            typename = typename std::enable_if<
                std::is_convertible<TypeUp*, Type*>::value>::type>
  aligned_delete(const aligned_delete<TypeUp>&) noexcept {}

  void operator()(Type* ptr) const {
    static_assert(sizeof(Type) > 0, "Can't delete pointer to incomplete type!");
    typedef typename std::remove_const<Type>::type TypeNonConst;
    Eigen::aligned_allocator<TypeNonConst> allocator;
    allocator.destroy(ptr);
    allocator.deallocate(ptr, 1u /*num*/);
  }
};
}  // namespace internal

template <typename Type>
using AlignedUniquePtr = std::unique_ptr<
    Type, internal::aligned_delete<typename std::remove_const<Type>::type>>;

template <typename Type, typename... Arguments>
inline AlignedUniquePtr<Type> aligned_unique(Arguments&&... arguments) {
  typedef typename std::remove_const<Type>::type TypeNonConst;
  Eigen::aligned_allocator<TypeNonConst> allocator;
  TypeNonConst* obj = allocator.allocate(1u);
  allocator.construct(obj, std::forward<Arguments>(arguments)...);
  return std::move(AlignedUniquePtr<Type>(obj));
}

}  // namespace dba
