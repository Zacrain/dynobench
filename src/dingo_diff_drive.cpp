#include "dynobench/dingo_diff_drive.hpp"
#include "dynobench/dyno_macros.hpp"
#include <fcl/geometry/shape/box.h>
#include <cmath>
#include <limits>

using Vxd = Eigen::VectorXd;

namespace dynobench {

void DingoDiffDrive_params::read_from_yaml(const char *file) {
  std::cout << "loading file: " << file << std::endl;
  filename = file;
  YAML::Node node = YAML::LoadFile(file);
  read_from_yaml(node);
}

void DingoDiffDrive_params::read_from_yaml(YAML::Node &node) {
  set_from_yaml(node, VAR_WITH_NAME(max_vel));
  set_from_yaml(node, VAR_WITH_NAME(min_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_angular_vel));
  set_from_yaml(node, VAR_WITH_NAME(min_angular_vel));
  set_from_yaml(node, VAR_WITH_NAME(shape));
  set_from_yaml(node, VAR_WITH_NAME(dt));
  set_from_yaml(node, VAR_WITH_NAME(size));
  set_from_yaml(node, VAR_WITH_NAME(distance_weights));
  set_from_yaml(node, VAR_WITH_NAME(tau_v));
  set_from_yaml(node, VAR_WITH_NAME(tau_w));
}

void DingoDiffDrive_params::write(std::ostream &out) {
  out << STR_(max_vel) << std::endl;
  out << STR_(min_vel) << std::endl;
  out << STR_(max_angular_vel) << std::endl;
  out << STR_(min_angular_vel) << std::endl;
  out << STR_(shape) << std::endl;
  out << STR_(dt) << std::endl;
  out << STR_(tau_v) << std::endl;
  out << STR_(tau_w) << std::endl;
  // size and distance_weights are vectors, so we need a different approach
  out << "size: ";
  for (size_t i = 0; i < size.size(); ++i) {
    out << size[i];
    if (i < size.size() - 1) out << ", ";
  }
  out << std::endl;
  out << "distance_weights: ";
  for (size_t i = 0; i < distance_weights.size(); ++i) {
    out << distance_weights[i];
    if (i < distance_weights.size() - 1) out << ", ";
  }
  out << std::endl;
}

Model_dingo_diff_drive::Model_dingo_diff_drive(const DingoDiffDrive_params &params,
                                               const Eigen::VectorXd &p_lb,
                                               const Eigen::VectorXd &p_ub)
    : Model_robot(std::make_shared<Rn>(5), 2), params(params) {

  double RM_low__ = -std::sqrt(std::numeric_limits<double>::max());
  double RM_max__ = std::sqrt(std::numeric_limits<double>::max());

  using V3d = Eigen::Vector3d;

  is_2d = true;
  nx_col = 3;  // collision checking uses (x, y, theta)
  nx_pr = 5;   // state space is (x, y, theta, v, w)
  translation_invariance = 2;

  distance_weights = Eigen::Map<const Eigen::VectorXd>(params.distance_weights.data(), params.distance_weights.size());
  name = "dingo_diff_drive";

  std::cout << "Robot name " << name << std::endl;
  std::cout << "Parameters" << std::endl;
  this->params.write(std::cout);
  std::cout << "***" << std::endl;

  ref_dt = params.dt;

  // State bounds: [x, y, theta, v, w]
  x_lb = Vxd::Constant(5, RM_low__);
  x_ub = Vxd::Constant(5, RM_max__);

  // Velocity bounds
  x_lb(3) = params.min_vel;
  x_ub(3) = params.max_vel;
  x_lb(4) = params.min_angular_vel;
  x_ub(4) = params.max_angular_vel;

  // Control bounds: [v_cmd, w_cmd] (velocity commands)
  u_lb = Vxd(2);
  u_ub = Vxd(2);
  u_lb << params.min_vel, params.min_angular_vel;
  u_ub << params.max_vel, params.max_angular_vel;

  // Create collision geometry
  if (params.shape == "box") {
    collision_geometries.push_back(
        std::make_shared<fcl::Boxd>(params.size.at(0), params.size.at(1), 0.1));
  } else {
    ERROR_WITH_INFO("shape not implemented: " + params.shape);
  }

  if (p_lb.size() && p_ub.size()) {
    set_position_lb(p_lb);
    set_position_ub(p_ub);
  }
}

Model_dingo_diff_drive::Model_dingo_diff_drive(const char *file,
                                               const Eigen::VectorXd &p_lb,
                                               const Eigen::VectorXd &p_ub)
    : Model_dingo_diff_drive([&] {
        DingoDiffDrive_params params;
        params.read_from_yaml(file);
        return params;
      }(), p_lb, p_ub) {}

void Model_dingo_diff_drive::step(Eigen::Ref<Eigen::VectorXd> xnext,
                                  const Eigen::Ref<const Eigen::VectorXd> &x,
                                  const Eigen::Ref<const Eigen::VectorXd> &u, double dt) {

  // State: [x, y, theta, v, w]
  // Control: [v_cmd, w_cmd] - velocity commands

  double x_pos = x(0);
  double y_pos = x(1);
  double theta = x(2);
  double v_current = x(3);
  double w_current = x(4);

  double v_cmd = u(0);
  double w_cmd = u(1);

  // Simple first-order dynamics: v_dot = (v_cmd - v_current) / tau
  // For realistic robot dynamics, we use a time constant approach
  // This models the fact that robots can't instantly change velocity
  double tau_v = params.tau_v;  // Linear velocity time constant (seconds)
  double tau_w = params.tau_w;  // Angular velocity time constant (seconds)

  // Compute velocity derivatives (accelerations)
  double v_dot = (v_cmd - v_current) / tau_v;
  double w_dot = (w_cmd - w_current) / tau_w;

  // Update velocities using first-order dynamics
  double v_new = v_current + v_dot * dt;
  double w_new = w_current + w_dot * dt;

  // Clamp velocities to bounds (safety)
  v_new = std::max(params.min_vel, std::min(params.max_vel, v_new));
  w_new = std::max(params.min_angular_vel, std::min(params.max_angular_vel, w_new));

  // Integrate kinematics using current velocities (mid-point integration for better accuracy)
  double v_avg = (v_current + v_new) * 0.5;
  double w_avg = (w_current + w_new) * 0.5;

  double x_new = x_pos + v_avg * cos(theta) * dt;
  double y_new = y_pos + v_avg * sin(theta) * dt;
  double theta_new = theta + w_avg * dt;

  // Normalize angle
  theta_new = atan2(sin(theta_new), cos(theta_new));

  xnext(0) = x_new;
  xnext(1) = y_new;
  xnext(2) = theta_new;
  xnext(3) = v_new;
  xnext(4) = w_new;
}

void Model_dingo_diff_drive::stepDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                                      Eigen::Ref<Eigen::MatrixXd> Ju,
                                      const Eigen::Ref<const Eigen::VectorXd> &x,
                                      const Eigen::Ref<const Eigen::VectorXd> &u, double dt) {

  double theta = x(2);
  double v_current = x(3);
  double w_current = x(4);
  double v_cmd = u(0);
  double w_cmd = u(1);

  // Time constants (same as in step function)
  double tau_v = params.tau_v;
  double tau_w = params.tau_w;

  // Initialize Jacobians
  Jx.setZero();
  Ju.setZero();

  // Compute intermediate values
  double v_dot = (v_cmd - v_current) / tau_v;
  double w_dot = (w_cmd - w_current) / tau_w;
  double v_new = v_current + v_dot * dt;
  double w_new = w_current + w_dot * dt;

  // For bounded velocities, we assume the bounds are not active (linearization around current point)
  // In practice, you might want to check if bounds are active and modify accordingly

  double v_avg = (v_current + v_new) * 0.5;
  double w_avg = (w_current + w_new) * 0.5;

  // d(x_new)/d(state) - Jacobian w.r.t. state
  Jx(0, 0) = 1.0;  // d(x_new)/d(x)
  Jx(1, 1) = 1.0;  // d(y_new)/d(y)
  Jx(2, 2) = 1.0;  // d(theta_new)/d(theta)

  // d(x_new)/d(theta)
  Jx(0, 2) = -v_avg * sin(theta) * dt;
  // d(y_new)/d(theta)
  Jx(1, 2) = v_avg * cos(theta) * dt;

  // d(x_new)/d(v_current) - position depends on average velocity
  double dv_avg_dv_current = 0.5 * (1.0 - dt / tau_v);
  Jx(0, 3) = dv_avg_dv_current * cos(theta) * dt;
  Jx(1, 3) = dv_avg_dv_current * sin(theta) * dt;

  // d(theta_new)/d(w_current) - orientation depends on average angular velocity
  double dw_avg_dw_current = 0.5 * (1.0 - dt / tau_w);
  Jx(2, 4) = dw_avg_dw_current * dt;

  // d(v_new)/d(v_current)
  Jx(3, 3) = 1.0 - dt / tau_v;
  // d(w_new)/d(w_current)
  Jx(4, 4) = 1.0 - dt / tau_w;

  // d(x_new)/d(control) - Jacobian w.r.t. control
  // d(x_new)/d(v_cmd)
  double dv_avg_dv_cmd = 0.5 * dt / tau_v;
  Ju(0, 0) = dv_avg_dv_cmd * cos(theta) * dt;
  Ju(1, 0) = dv_avg_dv_cmd * sin(theta) * dt;

  // d(theta_new)/d(w_cmd)
  double dw_avg_dw_cmd = 0.5 * dt / tau_w;
  Ju(2, 1) = dw_avg_dw_cmd * dt;

  // d(v_new)/d(v_cmd)
  Ju(3, 0) = dt / tau_v;
  // d(w_new)/d(w_cmd)
  Ju(4, 1) = dt / tau_w;
}

void Model_dingo_diff_drive::calcV(Eigen::Ref<Eigen::VectorXd> V,
                                   const Eigen::Ref<const Eigen::VectorXd> &x,
                                   const Eigen::Ref<const Eigen::VectorXd> &u) {
  // V is the "velocity" used for collision checking and visualization
  // For our 5D state, we return the velocity components

  double tau_v = params.tau_v;  // Same time constants as in step()
  double tau_w = params.tau_w;

//   // Ensure V has the right size
//   if (V.size() != 5) {
//     V.resize(5);
//   }

  // Check input dimensions
  assert(x.size() >= 5);
  assert(u.size() >= 2);

  V(0) = x(3) * cos(x(2));  // x_dot = v * cos(theta)
  V(1) = x(3) * sin(x(2));  // y_dot = v * sin(theta)
  V(2) = x(4);              // theta_dot = w
  V(3) = (u(0) - x(3)) / tau_v;  // v_dot = (v_cmd - v_current) / tau_v
  V(4) = (u(1) - x(4)) / tau_w;  // w_dot = (w_cmd - w_current) / tau_w
}

void Model_dingo_diff_drive::calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                                       Eigen::Ref<Eigen::MatrixXd> Jv_u,
                                       const Eigen::Ref<const Eigen::VectorXd> &x,
                                       const Eigen::Ref<const Eigen::VectorXd> &u) {
  // Compute Jacobians of calcV function
  // V(0) = x(3) * cos(x(2))  // x_dot = v * cos(theta)
  // V(1) = x(3) * sin(x(2))  // y_dot = v * sin(theta)
  // V(2) = x(4)              // theta_dot = w
  // V(3) = (u(0) - x(3)) / tau_v  // v_dot = (v_cmd - v_current) / tau_v
  // V(4) = (u(1) - x(4)) / tau_w  // w_dot = (w_cmd - w_current) / tau_w

  double tau_v = params.tau_v;
  double tau_w = params.tau_w;

  // Initialize matrices to zero (they should already be properly sized)
  Jv_x.setZero();
  Jv_u.setZero();

  // Check input dimensions
  assert(x.size() >= 5);
  assert(u.size() >= 2);

  double theta = x(2);
  double v = x(3);
  double w = x(4);

  // Jacobian w.r.t. state (Jv_x): dV/dx
  // dV(0)/dx: d(v*cos(theta))/dx
  Jv_x(0, 2) = -v * sin(theta);  // dV(0)/d(theta)
  Jv_x(0, 3) = cos(theta);       // dV(0)/d(v)

  // dV(1)/dx: d(v*sin(theta))/dx
  Jv_x(1, 2) = v * cos(theta);   // dV(1)/d(theta)
  Jv_x(1, 3) = sin(theta);       // dV(1)/d(v)

  // dV(2)/dx: d(w)/dx
  Jv_x(2, 4) = 1.0;              // dV(2)/d(w)

  // dV(3)/dx: d((u(0) - x(3)) / tau_v)/dx
  Jv_x(3, 3) = -1.0 / tau_v;     // dV(3)/d(v)

  // dV(4)/dx: d((u(1) - x(4)) / tau_w)/dx
  Jv_x(4, 4) = -1.0 / tau_w;     // dV(4)/d(w)

  // Jacobian w.r.t. control (Jv_u): dV/du
  // dV(3)/du: d((u(0) - x(3)) / tau_v)/du
  Jv_u(3, 0) = 1.0 / tau_v;      // dV(3)/d(v_cmd)

  // dV(4)/du: d((u(1) - x(4)) / tau_w)/du
  Jv_u(4, 1) = 1.0 / tau_w;      // dV(4)/d(w_cmd)
}

double Model_dingo_diff_drive::distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                                        const Eigen::Ref<const Eigen::VectorXd> &y) {
  assert(x.size() == 5);
  assert(y.size() == 5);

  // Compute weighted distance for our 5D state [x, y, theta, v, w]
  // Following the same pattern as unicycle2: [pos_L2, theta, v, w]
  // distance_weights = [1, 1, 0.5, 0.5] from YAML config
  
  Eigen::Vector4d raw_d = Eigen::Vector4d(
      (x.head<2>() - y.head<2>()).norm(),  // Position distance (L2 norm for x,y)
      std::abs(x(2) - y(2)),               // Angular difference (simple for Rn(5))
      std::abs(x(3) - y(3)),               // Linear velocity difference
      std::abs(x(4) - y(4)));              // Angular velocity difference

  return raw_d.dot(distance_weights);
}

int Model_dingo_diff_drive::number_of_r_dofs() {
  return 5;  // Full state dimension for dynamic model: x, y, theta, v, w
}

int Model_dingo_diff_drive::number_of_so2() {
  return 0;  // No SO(2) dimensions since we're using Rn(5) state space
}

int Model_dingo_diff_drive::number_of_robot() {
  return 1;  // Single robot
}

void Model_dingo_diff_drive::indices_of_so2(int &k, std::vector<size_t> &vect) {
  // No SO(2) indices since we're using Rn(5) state space
  // The yaw angle is treated as a regular Euclidean variable at index 2
}

} // namespace dynobench
