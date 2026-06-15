#include <fast_gicp/cuda/brute_force_knn.cuh>

#include <Eigen/Core>

#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

namespace fast_gicp {
  namespace cuda {

namespace {
  // Minimal in-place max-heap over a fixed-size buffer, ordered by pair::first
  // (the squared distance). Replaces nvbio::priority_queue, which is unmaintained
  // and does not compile with recent CUDA toolkits (CUDA 12.6).
  struct max_heap {
    __host__ __device__ explicit max_heap(thrust::pair<float, int>* data, int size = 0) : data(data), size(size) {}

    __host__ __device__ const thrust::pair<float, int>& top() const { return data[0]; }

    __host__ __device__ void push(const thrust::pair<float, int>& value) {
      int i = size++;
      while(i > 0) {
        const int parent = (i - 1) / 2;
        if(data[parent].first < value.first) {
          data[i] = data[parent];
          i = parent;
        } else {
          break;
        }
      }
      data[i] = value;
    }

    __host__ __device__ void pop() {
      const thrust::pair<float, int> last = data[--size];
      int i = 0;
      while(true) {
        int child = 2 * i + 1;
        if(child >= size) {
          break;
        }
        if(child + 1 < size && data[child].first < data[child + 1].first) {
          child++;
        }
        if(last.first < data[child].first) {
          data[i] = data[child];
          i = child;
        } else {
          break;
        }
      }
      data[i] = last;
    }

    thrust::pair<float, int>* data;
    int size;
  };

  struct neighborsearch_kernel {
    neighborsearch_kernel(int k, const thrust::device_vector<Eigen::Vector3f>& target, thrust::device_vector<thrust::pair<float, int>>& k_neighbors)
        : k(k), num_target_points(target.size()), target_points_ptr(target.data()), k_neighbors_ptr(k_neighbors.data()) {}

    template<typename Tuple>
    __host__ __device__ void operator()(const Tuple& idx_x) const {
      // threadIdx doesn't work because thrust split for_each in two loops
      int idx = thrust::get<0>(idx_x);
      const Eigen::Vector3f& x = thrust::get<1>(idx_x);

      // target points buffer & nn output buffer
      const Eigen::Vector3f* pts = thrust::raw_pointer_cast(target_points_ptr);
      thrust::pair<float, int>* k_neighbors = thrust::raw_pointer_cast(k_neighbors_ptr) + idx * k;

      // max-heap over the k nearest neighbors found so far
      max_heap queue(k_neighbors);

      for(int i = 0; i < k; i++) {
        float sq_dist = (pts[i] - x).squaredNorm();
        queue.push(thrust::make_pair(sq_dist, i));
      }

      for(int i = k; i < num_target_points; i++) {
        float sq_dist = (pts[i] - x).squaredNorm();
        if(sq_dist < queue.top().first) {
          queue.pop();
          queue.push(thrust::make_pair(sq_dist, i));
        }
      }
    }

    const int k;
    const int num_target_points;
    thrust::device_ptr<const Eigen::Vector3f> target_points_ptr;

    thrust::device_ptr<thrust::pair<float, int>> k_neighbors_ptr;
  };

  struct sorting_kernel {
    sorting_kernel(int k, thrust::device_vector<thrust::pair<float, int>>& k_neighbors) : k(k), k_neighbors_ptr(k_neighbors.data()) {}

    __host__ __device__ void operator()(int idx) const {
      // target points buffer & nn output buffer
      thrust::pair<float, int>* k_neighbors = thrust::raw_pointer_cast(k_neighbors_ptr) + idx * k;

      // the search kernel already left the k neighbors arranged as a max-heap
      max_heap queue(k_neighbors, k);

      for(int i = 0; i < k; i++) {
        thrust::pair<float, int> poped = queue.top();
        queue.pop();

        k_neighbors[k - i - 1] = poped;
      }
    }

    const int k;
    thrust::device_ptr<thrust::pair<float, int>> k_neighbors_ptr;
  };
}

void brute_force_knn_search(const thrust::device_vector<Eigen::Vector3f>& source, const thrust::device_vector<Eigen::Vector3f>& target, int k, thrust::device_vector<thrust::pair<float, int>>& k_neighbors, bool do_sort) {
  thrust::device_vector<int> d_indices(source.size());
  thrust::sequence(d_indices.begin(), d_indices.end());

  auto first = thrust::make_zip_iterator(thrust::make_tuple(d_indices.begin(), source.begin()));
  auto last = thrust::make_zip_iterator(thrust::make_tuple(d_indices.end(), source.end()));

  // k neighbor slots per source point
  k_neighbors.resize(source.size() * k, thrust::make_pair(-1.0f, -1));
  thrust::for_each(first, last, neighborsearch_kernel(k, target, k_neighbors));

  if(do_sort) {
    thrust::for_each(d_indices.begin(), d_indices.end(), sorting_kernel(k, k_neighbors));
  }
}

  }
} // namespace fast_gicp
