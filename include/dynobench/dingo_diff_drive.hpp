#pragma once

#include "dynobench/robot_models_base.hpp"

namespace dynobench {

struct DingoDiffDrive_params {
  void read_from_yaml(YAML::Node &node);
  void read_from_yaml(const char *file);
  void write(std::ostream &out);

  double max_vel = 1.3;
  double min_vel = -1.3;
  double max_angular_vel = 1.3 / 0.2405;
  double min_angular_vel = -1.3 / 0.2405;
  std::vector<double> size = {0.551, 0.517};
  std::vector<double> distance_weights = {1.0, 1.0, 0.5, 0.5};  // [position_L2, theta, linear_vel, angular_vel]
  std::string shape = "box";
  double dt = 0.05;
  double tau_v = 0.1;   // Linear velocity time constant (seconds)
  double tau_w = 0.05;  // Angular velocity time constant (seconds)
  std::string filename = "";
};

class Model_dingo_diff_drive : public Model_robot {

public:
  virtual ~Model_dingo_diff_drive() = default;

  // Constructor from parameters
  Model_dingo_diff_drive(const DingoDiffDrive_params &params,
                         const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
                         const Eigen::VectorXd &p_ub = Eigen::VectorXd());

  // Constructor from file
  Model_dingo_diff_drive(const char *file,
                         const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
                         const Eigen::VectorXd &p_ub = Eigen::VectorXd());

  virtual void step(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u, double dt) override;

  virtual void stepDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u, double dt) override;

  virtual void calcV(Eigen::Ref<Eigen::VectorXd> V,
                     const Eigen::Ref<const Eigen::VectorXd> &x,
                     const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual void calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                         Eigen::Ref<Eigen::MatrixXd> Jv_u,
                         const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual double distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                          const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual int number_of_r_dofs() override;
  virtual int number_of_so2() override;
  virtual void indices_of_so2(int &k, std::vector<size_t> &vect) override;
  virtual int number_of_robot() override;

  DingoDiffDrive_params params;
};

} // namespace dynobench
