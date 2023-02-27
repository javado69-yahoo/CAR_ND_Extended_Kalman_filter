#include "kalman_filter.h"
#include "tools.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;


/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  Tools tools_;
  MatrixXd Hj_ = tools_.CalculateJacobian(x_);
    if (Hj_.isZero(0))
    {
        return;
    }
    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);

    VectorXd c_to_p(3);
    float dist = sqrt(px * px + py * py);
    float angle = atan2(py, px);
    float angular_vel = (px * vx + py * vy) / dist;
    c_to_p << dist, angle, angular_vel;
    VectorXd y = z - c_to_p;
    while (y(1) > M_PI)
    {
        y(1) -= 2 * M_PI;
    }
    while (y(1) <= -M_PI)
    {
        y(1) += 2 * M_PI;
    }

    MatrixXd S = Hj_ * P_ * Hj_.transpose() + R_;
    MatrixXd K = P_ * Hj_.transpose() * S.inverse();

    x_ += (K * y);
    MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
    P_ = (I - K * Hj_) * P_;
}
