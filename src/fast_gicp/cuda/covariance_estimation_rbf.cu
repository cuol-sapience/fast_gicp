#include <fast_gicp/cuda/covariance_estimation.cuh>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

namespace fast_gicp {
namespace cuda {

struct NormalDistribution {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  __host__ __device__ NormalDistribution() {}

  static __host__ __device__ NormalDistribution zero() {
    NormalDistribution dist;
    dist.sum_weights = 0.0f;
    dist.mean.setZero();
    dist.cov.setZero();
    return dist;
  }

  __host__ __device__ NormalDistribution operator+(const NormalDistribution& rhs) const {
    NormalDistribution sum;
    sum.sum_weights = sum_weights + rhs.sum_weights;
    sum.mean = mean + rhs.mean;
    sum.cov = cov + rhs.cov;
    return sum;
  }

  __host__ __device__ NormalDistribution& operator+=(const NormalDistribution& rhs) {
    sum_weights += rhs.sum_weights;
    mean += rhs.mean;
    cov += rhs.cov;
    return *this;
  }

  __host__ __device__ void accumulate(const float w, const Eigen::Vector3f& x) {
    sum_weights += w;
    mean += w * x;
    cov += w * x * x.transpose();
  }

  __host__ __device__ NormalDistribution& finalize() {
    Eigen::Vector3f sum_pt = mean;
    mean /= sum_weights;
    cov = (cov - mean * sum_pt.transpose()) / sum_weights;

    return *this;
  }

  float sum_weights;
  Eigen::Vector3f mean;
  Eigen::Matrix3f cov;
};

struct covariance_estimation_kernel {
  static const int BLOCK_SIZE = 512;

  covariance_estimation_kernel(thrust::device_ptr<const float> exp_factor_ptr, thrust::device_ptr<const float> max_dist_ptr, thrust::device_ptr<const Eigen::Vector3f> points_ptr)
  : exp_factor_ptr(exp_factor_ptr),
    max_dist_ptr(max_dist_ptr),
    points_ptr(points_ptr) {}

  __host__ __device__ NormalDistribution operator()(const Eigen::Vector3f& x) const {
    const float exp_factor = *thrust::raw_pointer_cast(exp_factor_ptr);
    const float max_dist = *thrust::raw_pointer_cast(max_dist_ptr);
    const float max_dist_sq = max_dist * max_dist;
    const Eigen::Vector3f* points = thrust::raw_pointer_cast(points_ptr);

    NormalDistribution dist = NormalDistribution::zero();
    for (int i = 0; i < BLOCK_SIZE; i++) {
      float sq_d = (x - points[i]).squaredNorm();
      if (sq_d > max_dist_sq) {
        continue;
      }

      float w = expf(-exp_factor * sq_d);
      dist.accumulate(w, points[i]);
    }

    return dist;
  }

  thrust::device_ptr<const float> exp_factor_ptr;
  thrust::device_ptr<const float> max_dist_ptr;
  thrust::device_ptr<const Eigen::Vector3f> points_ptr;
};

struct finalize_cov_kernel {
  __host__ __device__ Eigen::Matrix3f operator()(NormalDistribution dist) const {
    return dist.finalize().cov;
  }
};

void covariance_estimation_rbf(const thrust::device_vector<Eigen::Vector3f>& points, double kernel_width, double max_dist, thrust::device_vector<Eigen::Matrix3f>& covariances) {
  covariances.resize(points.size());

  thrust::device_vector<float> constants(2);
  constants[0] = kernel_width;
  constants[1] = max_dist;
  thrust::device_ptr<const float> exp_factor_ptr = constants.data();
  thrust::device_ptr<const float> max_dist_ptr = constants.data() + 1;

  int num_blocks = (points.size() + (covariance_estimation_kernel::BLOCK_SIZE - 1)) / covariance_estimation_kernel::BLOCK_SIZE;
  // padding
  thrust::device_vector<Eigen::Vector3f> ext_points(num_blocks * covariance_estimation_kernel::BLOCK_SIZE);
  thrust::copy(points.begin(), points.end(), ext_points.begin());
  thrust::fill(ext_points.begin() + points.size(), ext_points.end(), Eigen::Vector3f(0.0f, 0.0f, 0.0f));

  // Running sum per point: O(n) peak memory instead of O(n * num_blocks).
  // Previously accumulated_dists held all num_blocks slices simultaneously,
  // which grew to ~3 GB for large maps (OOM on Jetson Orin Nano 8 GB).
  thrust::device_vector<NormalDistribution> running_dists(points.size(), NormalDistribution::zero());
  thrust::device_vector<NormalDistribution> block_dists(points.size());

  for (int i = 0; i < num_blocks; i++) {
    covariance_estimation_kernel kernel(exp_factor_ptr, max_dist_ptr, ext_points.data() + covariance_estimation_kernel::BLOCK_SIZE * i);
    thrust::transform(thrust::cuda::par, points.begin(), points.end(), block_dists.begin(), kernel);
    thrust::transform(
      thrust::cuda::par,
      running_dists.begin(), running_dists.end(),
      block_dists.begin(),
      running_dists.begin(),
      thrust::plus<NormalDistribution>());
  }

  thrust::transform(thrust::cuda::par, running_dists.begin(), running_dists.end(), covariances.begin(), finalize_cov_kernel());
}

}  // namespace cuda
}  // namespace fast_gicp