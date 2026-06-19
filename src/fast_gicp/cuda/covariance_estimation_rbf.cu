#include <fast_gicp/cuda/covariance_estimation.cuh>

#include <thrust/device_vector.h>
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

  __host__ __device__ void accumulate(const float w, const Eigen::Vector3f& x) {
    sum_weights += w;
    mean += w * x;
    cov += w * x * x.transpose();
  }

  __host__ __device__ Eigen::Matrix3f finalize() {
    Eigen::Vector3f sum_pt = mean;
    mean /= sum_weights;
    return (cov - mean * sum_pt.transpose()) / sum_weights;
  }

  float sum_weights;
  Eigen::Vector3f mean;
  Eigen::Matrix3f cov;
};

// One thread per source point; iterates ALL target points in a single GPU kernel.
// O(n) memory, O(1) kernel launches — avoids the host-side loop overhead of the
// old BLOCK_SIZE streaming approach while keeping the same O(n^2) compute.
struct covariance_estimation_rbf_kernel {
  covariance_estimation_rbf_kernel(float exp_factor, float max_dist_sq, const Eigen::Vector3f* neighbors, int n_neighbors)
  : exp_factor(exp_factor), max_dist_sq(max_dist_sq), neighbors(neighbors), n_neighbors(n_neighbors) {}

  __device__ Eigen::Matrix3f operator()(const Eigen::Vector3f& x) const {
    NormalDistribution dist = NormalDistribution::zero();
    for (int i = 0; i < n_neighbors; i++) {
      const float sq_d = (x - neighbors[i]).squaredNorm();
      if (sq_d > max_dist_sq) {
        continue;
      }
      dist.accumulate(__expf(-exp_factor * sq_d), neighbors[i]);
    }
    return dist.finalize();
  }

  float exp_factor;
  float max_dist_sq;
  const Eigen::Vector3f* neighbors;
  int n_neighbors;
};

void covariance_estimation_rbf(const thrust::device_vector<Eigen::Vector3f>& points, double kernel_width, double max_dist, thrust::device_vector<Eigen::Matrix3f>& covariances) {
  covariances.resize(points.size());

  const float exp_factor = static_cast<float>(kernel_width);
  const float max_dist_sq = static_cast<float>(max_dist * max_dist);
  const Eigen::Vector3f* pts = thrust::raw_pointer_cast(points.data());

  covariance_estimation_rbf_kernel kernel(exp_factor, max_dist_sq, pts, static_cast<int>(points.size()));
  thrust::transform(thrust::cuda::par, points.begin(), points.end(), covariances.begin(), kernel);
}

}  // namespace cuda
}  // namespace fast_gicp
