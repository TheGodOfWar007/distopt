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

/// @brief definition of dba (distributed bundle adjustment) namespace
namespace dba {

/// @brief structure returning a pair of hashed values
struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

/// @brief compute key given that id_a is lower than id_b
/// @param id_a 
/// @param id_b 
/// @return unique directed key
inline auto computeDirectedKeyFromIds(const uint64_t id_a, const uint64_t id_b)
    -> uint64_t {
  CHECK_LT(id_a + 1, std::numeric_limits<uint64_t>::max() / key_multiplier);
  return (id_a + 1) * key_multiplier + id_b;
}

/// @brief compute key given id_a and id_b
/// @param id_a 
/// @param id_b 
/// @return computed key
inline auto computeKeyFromIds(const uint64_t id_a, const uint64_t id_b)
    -> uint64_t {
  const auto id_lo = (id_a <= id_b) ? id_a + 1 : id_b + 1;
  const auto id_hi = (id_a > id_b) ? id_a + 1 : id_b + 1;
  CHECK_LT(id_lo, std::numeric_limits<uint64_t>::max() / key_multiplier);
  return id_lo * key_multiplier + id_hi;
}

/// @brief given a key, compute the pair of ids that it represents
/// @param key 
/// @return pair of ids
inline auto computeIdsFromKey(const uint64_t key)
    -> std::pair<uint64_t, uint64_t> {
  const auto id_lo = (key / key_multiplier) - 1;
  const auto id_hi = (key % key_multiplier) - 1;
  return {id_lo, id_hi};
}

/// @brief observation struct
struct Observation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector2d obs;  // 2d Eigen vector describing obersvation
  uint64_t mp_id;       // 3D Map Point ID
  uint64_t frame_id;    // observation frame
};

using ObservationVector =
    std::vector<Observation, Eigen::aligned_allocator<Observation>>;
using ObservationMap = std::unordered_map<
    uint64_t, Observation, std::hash<uint64_t>, std::equal_to<uint64_t>,
    Eigen::aligned_allocator<std::pair<uint64_t, Observation>>>;

/// @brief define a PoseDual struct template
/// @tparam NI non-type template parameter describing number of intrinsic parameters
/// @tparam ND non-type template parameter descrbing number of distortion parameters
/// @details PoseDual struct template contains an ID and a vars_ array containing position, rotation, intrinsics parameters, and distortion parameters
template <size_t NI, size_t ND>
struct PoseDual {
  /// @brief default PoseDual constructor
  PoseDual() : id_(std::numeric_limits<uint64_t>::max()) { fill(0.0); }       // define id as the max unsigned 64-bit int
  /// @brief PoseDual constructor with custom id
  /// @param id custom id for PoseDual
  PoseDual(const uint64_t id) : id_(id) { fill(0.0); }

  /// @brief overload to allow access to elements in vars_ array
  /// @param i index
  /// @return parameter indexed
  auto operator[](const size_t i) -> double& { return vars_[i]; }
  auto operator[](const size_t i) const -> const double& { return vars_[i]; }

  auto setId(const uint64_t id) -> void { id_ = id; }                           // set PosDual Id
  auto getId() const -> uint64_t { return id_; }                                // get PosDual Id

  /// @brief give pointer to position
  /// @return pointer to beginning of vars_ array
  auto getPosition() -> double* { return vars_; }
  /// @brief return pointer to rotation
  /// @return pointer to location in vars_ array where rotation elements are stored
  auto getRotation() -> double* { return vars_ + kNumPosParams; }
  /// @brief return pointer to intrinsic parameters
  /// @return pointer to location in vars_ array where intrinsic parameters are stored
  auto getIntrisincs() -> double* {
    return vars_ + kNumPosParams + kNumRotParams;
  }
  /// @brief return pointer to distortion parameters
  /// @return pointer to location in vars_ array where distortion parameters are stored
  auto getDistortion() -> double* {
    return vars_ + kNumPosParams + kNumRotParams + NI;
  }
  auto fill(const double val) -> void {
    for (auto& v : vars_) v = val;
  }
  /// @brief get size of pose dual
  /// @return return [3+3=6] pos & rot params + NI + ND
  static constexpr auto getSize() -> size_t {
    return kNumPosParams + kNumRotParams + NI + ND;
  }

 private:
  static constexpr size_t kNumPosParams = 3; // pos defined as xyz
  static constexpr size_t kNumRotParams = 3; // rot defined as rotatio about 3 axes

 private:
  uint64_t id_;
  // array of size kNumPosParams + kNumRotParams + NI + ND
  double vars_[kNumPosParams + kNumRotParams + NI + ND];
};


/// @brief struct definition for MapPointDual
/// @details a MapPointDual is describe as a 3D point in space's dual variable
struct MapPointDual {
  /// @brief default MapPointDual constructor setting id to maximum 64-bit unsigned integer
  MapPointDual() : id_(std::numeric_limits<uint64_t>::max()) {}
  /// @brief MapPointDual constructor with constum id
  /// @param id custom MapPointDual id
  MapPointDual(const uint64_t id) : id_(id) {}

  /// @brief overload to allow access to elements in vars_ array
  /// @param i index
  /// @return parameter indexed
  auto operator[](const size_t i) -> double& { return vars_[i]; }
  auto operator[](const size_t i) const -> const double& { return vars_[i]; }

  auto setId(const uint64_t id) -> void { id_ = id; }
  auto getId() const -> uint64_t { return id_; }

  /// @brief give pointer to coordinates of 3D point
  /// @return pointer to beginning of vars_ array
  auto getPosition() -> double* { return vars_; }
  /// @brief fill vars_ with input value val
  /// @param val input value to fill vars_ with
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

/// @brief alias of a vector of Eigen Vector3 objects holding doubles
using VectorOfVector3 =
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;

/// @brief alias of a vector of Eigen Matrices of size Nx1 holding doubles
/// @tparam N arbitrary desired fixed Vector size
template <int N>
using VectorOfVectorN =
    std::vector<Eigen::Matrix<double, N, 1>,
                Eigen::aligned_allocator<Eigen::Matrix<double, N, 1>>>;

/// @brief namespace internal : Stolen from ASLAM in order to be able to use uniqe
namespace internal {
/// @brief custom deleter to delete dynamically allocated memory for objects of a type Type
/// @details delete memory dynamically allocated using Eigen::aligned_allocator for objects of templated type Type
/// @tparam Type the type of desired deleted memory
template <typename Type>
struct aligned_delete {
  constexpr aligned_delete() noexcept = default;

  template <typename TypeUp,
            typename = typename std::enable_if<
                std::is_convertible<TypeUp*, Type*>::value>::type>
  aligned_delete(const aligned_delete<TypeUp>&) noexcept {}

  /// @brief take pointer to object of type Type and delete the object
  /// @param ptr pointer to object of type Type
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
