#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <thread>
#include <vector>
#include <zmq.hpp>

#include "pose.hpp"
#include "calibration_guard.hpp"
#include "feedback_clock.hpp"
#include "spline.hpp"
#include "ur_rtde/dashboard_client.h"
#include "ur_rtde/rtde_control_interface.h"
#include "ur_rtde/rtde_receive_interface.h"

using json = nlohmann::json;

inline std::stringstream t() {
    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    auto now_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << "[" << std::put_time(std::localtime(&now_time), "%H:%M:%S") << "." << std::setfill('0')
       << std::setw(3) << now_ms.count() << "] ";
    return ss;
}

class PositionAccelerationLimiter {
   public:
    PositionAccelerationLimiter(const double max_accel, const double frequency)
        : max_accel(max_accel), frequency(frequency) {}

    void apply_position(Pose7D &current_velocity) {
        Eigen::Vector<double, 3> acceleration =
            (current_velocity.position - previous_velocity.position) * frequency;
        acceleration = acceleration.cwiseMin(max_accel).cwiseMax(-max_accel);

        Eigen::Vector<double, 3> limited_velocity =
            previous_velocity.position + acceleration / frequency;
        current_velocity.position = limited_velocity;
        previous_velocity = current_velocity;
    }

   private:
    const double max_accel;
    const double frequency;
    Pose7D previous_velocity;
};

enum class ControllerMode { Idle, JointTarget, EEPoseTarget, ReferencePath };

class UR5Controller {
   public:
    UR5Controller(const std::string &robot_ip, const json &config)
        : context(1),
          socket(context, ZMQ_SUB),
          publisher(context, ZMQ_PUB), 
          calibration_guard(config.value("calibration_safety", json(nullptr))),
          control(robot_ip),
          receive(robot_ip),
          dashboard_client(robot_ip),
          stop_flag(false) {

        // general parameters
        frequency = config["frequency"].get<double>();
        reset_pose = config["reset_pose"].get<std::vector<double>>();
        verbose = config["verbose"].get<bool>();

        // ee-control parameters
        kpd_values = config["kpd_values"].get<std::vector<double>>();
        p_delta = config["position_delta"].get<double>();
        o_delta = config["orientation_delta"].get<double>();
        exp_filter = config["exponential_filter"].get<double>();
        position_acceleration = config["position_acceleration"].get<double>();
        orientation_acceleration = config["orientation_acceleration"].get<double>();
        deadspace_speed = config["deadspace_speed"].get<double>();

        // trajectory tracking parameters
        max_joint_velocity = config["max_joint_velocity"].get<double>();
        joint_acceleration = config["joint_acceleration"].get<double>();
        path_tracking_p_gain = config["path_tracking_p_gain"].get<double>();
        joint_command_timeout_s = config.value("joint_command_timeout_s", 0.2);
        if (joint_command_timeout_s <= 0.0) {
            throw std::invalid_argument("joint_command_timeout_s must be positive");
        }

        const std::string socket_address =
            "tcp://127.0.0.1:" + std::to_string(config["socket_port"].get<int>());
        const std::string publisher_address =
            "tcp://127.0.0.1:" + std::to_string(config["publisher_port"].get<int>());
        socket.set(zmq::sockopt::subscribe, "");  // Subscribe to all messages
        socket.set(zmq::sockopt::conflate, 1);socket.connect(socket_address);
        
        std::cout << "Connecting to sockets" << std::endl;
        std::cout << socket_address << std::endl;
        std::cout << publisher_address << std::endl;
        socket.connect(socket_address);
        publisher.bind(publisher_address);

        std::cout << "Connecting to robot at " << robot_ip << std::endl;
        dashboard_client.connect();

        std::cout << "started UR5 Controller on ip " << robot_ip << " with frequency " << frequency
                  << " Hz\n";
        std::cout << "Joint target command timeout: " << joint_command_timeout_s
                  << " s (stale target_q stream triggers speedStop)\n";
    }

    Pose7D calculate_speed(const Pose7D &oldSpeed) {
        // position
        pose_diff = target_pose - current_pose;
        const double norm = pose_diff.position.norm();
        if (norm > p_delta) {
            pose_diff.position = pose_diff.position / norm * p_delta;
        }

        pose_vel_diff = eigen_to_std(-curr_vel);
        ee_speed.position =
            kpd_values[0] * pose_diff.position + kpd_values[1] * pose_vel_diff.position;

        // orientation
        const double angle = pose_diff.get_rotation_angle();
        if (angle > o_delta) {
            // if zero, all Identity, if 1, all diff.quat
            pose_diff.quat = Eigen::Quaterniond::Identity().slerp(o_delta / angle, pose_diff.quat);
        }

        ee_speed.quat = pose_diff.scale_and_return_quat(kpd_values[2]) *
                     pose_vel_diff.scale_and_return_quat(kpd_values[3]);

        // downward force adaption
        // if (curr_force[2] > 3.5 and ee_speed.position[2] < 0.) {
        //     ee_speed.position[2] = std::max((1.5 - curr_force_lowpass[2]), 0.) * ee_speed.position[2] +
        //                         std::min(curr_force_lowpass[2] - 0.5, 1.) * 20.;
        // }

        apply_exp_filter(ee_speed, oldSpeed, exp_filter);
        return ee_speed;
    }

    void update_robot_state() {
        rtde_clock.observe(receive.getTimestamp());
        if (!rtde_clock.healthy(receive.isConnected())) {
            rtde_feedback_failed = true;
            if (calibration_guard.enabled)
                calibration_guard.trip("Robot RTDE feedback stalled/reset/disconnected; restart required");
        }
        current_pose = receive.getActualTCPPose();
        curr_vel = utils::std_to_eigen(receive.getActualTCPSpeed());
        curr_Q = utils::std_to_eigen(receive.getActualQ());
        curr_Qd = utils::std_to_eigen(receive.getActualQd());
        curr_force = utils::std_to_eigen(receive.getActualTCPForce());
        curr_force_lowpass = 0.9 * curr_force_lowpass + 0.1 * curr_force;
    }

    int get_reference() {
        zmq::message_t request;
        if (socket.recv(request, zmq::recv_flags::dontwait)) {
            try {
                std::string received_message(static_cast<char *>(request.data()), request.size());
                const auto received_json = json::parse(received_message);

                if (received_json.value("stop", false)) {
                    stop_requested = true;
                    return 4;
                }
                if (calibration_guard.enabled) {
                    calibration_guard.update_hand(received_json);
                    if (received_json.contains("calibration_arm")) {
                        calibration_guard.arm(received_json.at("calibration_arm").get<std::string>(),
                            current_pose.to_std_vector(), utils::eigen_to_std(curr_Q));
                        stop_requested = true;
                        return 0;
                    }
                    CalibrationGuard::require(received_json.contains("target_ee_pose") && received_json.contains("calibration_session"),
                        "Only session-bound calibration TCP commands allowed in guarded mode");
                    const auto pose = received_json.at("target_ee_pose").get<std::vector<double>>();
                    calibration_guard.target(received_json.at("calibration_session").get<std::string>(), pose, CalibrationGuard::Clock::now());
                    target_pose = pose;
                    current_mode = ControllerMode::EEPoseTarget;
                    return 1;
                }
                if (received_json.contains("target_pose")) {
                    const std::vector<double> pose_vector =
                        received_json["target_pose"].get<std::vector<double>>();
                    target_pose = pose_vector;
                    std::cout << t().str() << "Received target pose: ["
                              << target_pose.position.transpose() << ", "
                              << target_pose.quat.coeffs().transpose() << "]\n";
                    std::cout
                        << "WARNING: This name will be deprecated, please use 'target_ee_pose'"
                        << std::endl;

                    current_mode = ControllerMode::EEPoseTarget;
                    joint_command_stream_active = false;
                    return 1;
                } else if (received_json.contains("target_ee_pose")) {
                    const std::vector<double> pose_vector =
                        received_json["target_ee_pose"].get<std::vector<double>>();
                    target_pose = pose_vector;
                    std::cout << t().str() << "Received target pose: ["
                              << target_pose.position.transpose() << ", "
                              << target_pose.quat.coeffs().transpose() << "]\n";

                    current_mode = ControllerMode::EEPoseTarget;
                    joint_command_stream_active = false;
                    return 1;
                } else if (received_json.contains("reset_joint_angles")) {
                    target_Q = utils::std_to_eigen(received_json["reset_joint_angles"]);
                    std::cout << t().str() << "Received reset joint angles: ["
                              << target_Q.transpose() << "]\n";
                    std::cout << "WARNING: This name will be deprecated, please use 'target_q'"
                              << std::endl;

                    current_mode = ControllerMode::JointTarget;
                    last_joint_command_time = std::chrono::steady_clock::now();
                    joint_command_stream_active = true;
                    return 2;
                } else if (received_json.contains("target_q")) {
                    target_Q = utils::std_to_eigen(received_json["target_q"]);
                    if (verbose) {
                        std::cout << t().str() << "Received streamed joint target (rad): ["
                                  << target_Q.transpose() << "]\n";
                    }

                    current_mode = ControllerMode::JointTarget;
                    last_joint_command_time = std::chrono::steady_clock::now();
                    joint_command_stream_active = true;
                    return 2;
                } else if (received_json.value("stop", false)) {
                    stop_requested = true;
                    return 4;
                } else if(received_json.contains("reset_force_sensor")){
                    reset_force_sensor = true;
                    std::cout << t().str() << "Received reset force sensor command "
                              << received_json["reset_force_sensor"] << "\n";
                    return 3;                    
                } else if (received_json.contains("path") && received_json.contains("time")) {
                    std::cout << t().str() << "Received a reference path. \n";

                    const auto times = received_json["time"].get<std::vector<double>>();
                    const auto poses =
                        received_json["path"].get<std::vector<std::vector<double>>>();

                    if (times.size() != poses.size()) {
                        std::cout << "WARNING: length of times and poses do not match."
                                  << std::endl;
                    }

                    std::vector<Eigen::VectorXd> eigen_poses;
                    eigen_poses.reserve(poses.size());  // Pre-allocate memory for efficiency

                    for (const auto &pose : poses) {
                        Eigen::VectorXd eigen_pose(pose.size());
                        for (size_t i = 0; i < pose.size(); ++i) {
                            eigen_pose(i) = pose[i];
                        }
                        eigen_poses.push_back(eigen_pose);
                    }

                    // TODO: make sure this works
                    Eigen::VectorXd v0 = curr_vel;

                    // make spline
                    ref_path = TkSpline(times, eigen_poses, v0);

                    // TODO: move to function
                    // TODO: do ee & joint coordinate check
                    const uint num_pts_to_check = 100;
                    for (uint i = 0; i < num_pts_to_check; ++i) {
                        const double t = 1. * i / num_pts_to_check * times.back();
                        const Eigen::VectorXd velocity = ref_path.get_velocity(t);

                        if (velocity.cwiseAbs().maxCoeff() > max_joint_velocity) {
                            std::cerr << "WARNING: too large velocity at time " << t << " ("
                                      << velocity.cwiseAbs().maxCoeff() << ")" << std::endl;
                            // return -1;
                        }
                    }

                    path_tracking_start_time = std::chrono::system_clock::now();
                    current_mode = ControllerMode::ReferencePath;
                    joint_command_stream_active = false;
                    return 3;
                } else {
                    std::cerr << t().str()
                              << "Invalid JSON format: Missing 'target_pose' or "
                                 "'reset_joint_angles' field\n";
                    return -1;
                }
            } catch (const std::exception &e) {
                if (calibration_guard.enabled) { calibration_guard.trip(e.what()); stop_requested = true; }
                std::cerr << t().str() << "Error parsing JSON: " << e.what() << "\n";
                return -1;
            }
        }
        return 0;
    }

    bool reconnect() {
        if (receive.isProtectiveStopped() || receive.isEmergencyStopped()) {
            std::cerr << "Protective/emergency stop: inspect the cause and recover manually.\n";
            return false;
        }

        for (int i = 0; i < 10; ++i) {
            if (stop_flag.load()) {
                std::cout << "Reconnection attempt aborted\n";
                return false;
            }
            try {
                control.disconnect();
                control.reconnect();
                std::cout << "Reconnected to UR Controller\n";
                return true;
            } catch (const std::exception &e) {
                std::cerr << "Reconnection attempt " << (i + 1) << " failed: " << e.what() << "\n";
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }
        return false;
    }

    void send_state() {
        update_robot_state();
        const json state = {
            {"pos", current_pose.to_std_vector()},
            {"vel", utils::eigen_to_std(curr_vel)},
            {"Q", utils::eigen_to_std(curr_Q)},
            {"Qd", utils::eigen_to_std(curr_Qd)},
            {"force", utils::eigen_to_std(curr_force_lowpass)},
            {"is_truncated", !success || rtde_feedback_failed},
            {"robot_state_age_s", rtde_clock.age()},
            {"rtde_connected", receive.isConnected()},
            {"protective_stopped", receive.isProtectiveStopped()},
            {"emergency_stopped", receive.isEmergencyStopped()},
            {"calibration_guard", calibration_guard.status()},
            {"joint_target_watchdog_s", joint_command_timeout_s},
            {"timestamp_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::system_clock::now().time_since_epoch())
                                 .count()}};

        const std::string message = state.dump();
        zmq::message_t zmq_message(message.begin(), message.end());
        publisher.send(zmq_message, zmq::send_flags::none);
    }

    bool stop_motion(const std::string &reason) {
        const bool stopped = control.speedStop(joint_acceleration);
        target_Q.resize(0);
        joint_command_stream_active = false;
        stop_requested = false;
        calibration_guard.moving = false;
        guard_previous_velocity.setZero();
        current_mode = ControllerMode::Idle;
        prev_mode = ControllerMode::Idle;
        applied_speed.position.setZero();
        applied_speed.quat = Eigen::Quaterniond::Identity();
        update_robot_state();
        target_pose = current_pose.copy();

        if (stopped) {
            std::cout << t().str() << "UR5 speedStop applied: " << reason << "\n";
        } else {
            std::cerr << t().str() << "UR5 speedStop failed: " << reason << "\n";
        }
        return stopped;
    }

    void run() {
        update_robot_state();
        target_pose = current_pose.copy();
        control.zeroFtSensor();
        control.setWatchdog(calibration_guard.enabled ? 10.0 : 0.1);

        PositionAccelerationLimiter limiter(position_acceleration, frequency);

        while (!stop_flag.load()) {
            const auto start = std::chrono::high_resolution_clock::now();

            get_reference();

            if (calibration_guard.enabled && (!success || receive.isProtectiveStopped() || receive.isEmergencyStopped())) {
                calibration_guard.trip("Robot stop/RTDE failure; manual inspection and controller restart required");
                stop_motion(calibration_guard.fault);
                return;
            }
            if (!success) {
                std::cout << "\nUR Controller not connected anymore, attempting to reconnect\n";
                send_state();
                if (!reconnect()) {
                    return;
                } else {
                    std::cout << "Reconnected! Should the robot arm move to the reset pose (y), continue without resetting (c), or quit (q/n)? ";
                    char user_input;
                    std::cin >> user_input;

                    if (user_input == 'q' || user_input == 'n') {
                        stop_flag.store(true);
                        return;
                    } else if (user_input == 'c') {
                        std::cout << "Continuing without resetting the pose...\n";
                    } else if (user_input == 'y') {
                        control.moveJ(reset_pose);
                        control.zeroFtSensor();
                        update_robot_state();
                        target_pose = current_pose.copy();
                    } else {
                        std::cerr << "Invalid input. Continuing without resetting the pose...\n";
                    }
                    success = true;
                    continue;
                }
            }

            control.kickWatchdog();
            send_state();
            if (rtde_feedback_failed) {
                stop_motion("Robot RTDE feedback stalled/reset/disconnected; restart required");
                send_state();
                return;
            }
            if (calibration_guard.enabled && calibration_guard.fault.empty()) {
                try {
                    CalibrationGuard::require(receive.isConnected() && rtde_clock.age() <= calibration_guard.value("state_timeout_s"),
                        "Robot RTDE feedback stalled/disconnected");
                    calibration_guard.tick(current_pose.to_std_vector(), utils::eigen_to_std(curr_Q),
                        utils::eigen_to_std(curr_vel), utils::eigen_to_std(curr_Qd), CalibrationGuard::Clock::now());
                } catch (const std::exception &e) {
                    calibration_guard.trip(e.what());
                    stop_requested = true;
                }
            }
            if (calibration_guard.enabled && !calibration_guard.fault.empty() && current_mode != ControllerMode::Idle)
                stop_requested = true;

            if (stop_requested) {
                success &= stop_motion("explicit stop command");
                continue;
            }

            if (current_mode == ControllerMode::JointTarget &&
                joint_command_stream_active) {
                const double command_age_s =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - last_joint_command_time)
                        .count();
                if (command_age_s > joint_command_timeout_s) {
                    success &= stop_motion(
                        "target_q stream timed out after " +
                        std::to_string(command_age_s) + " s");
                    continue;
                }
            }

            if (reset_force_sensor){
                control.zeroFtSensor();
                reset_force_sensor = false;
            }

            // Track a streamed joint target; this is not a reset/home command.
            if (current_mode == ControllerMode::JointTarget) {
                if (target_Q.size() > 0) {
                    if (target_Q.size() != 6) {
                        std::cerr << "Invalid target_Q size: Expected 6 elements, got "
                                  << target_Q.size() << "\n";
                    } else {
                        if (prev_mode != ControllerMode::JointTarget){
                            success &= control.speedStop(3.0);
                        }
                        // std::this_thread::sleep_for(std::chrono::milliseconds(20));

                        control.setWatchdog(0.001);  // temporarily disable watchdog
                        // success &= control.moveJ(eigen_to_std(target_Q), 1.5, 0.7);
                        const Eigen::VectorXd desired_joint_speed =
                            path_tracking_p_gain * (target_Q - curr_Q);

                        const Eigen::VectorXd clamped_joint_speed = utils::clamp(
                            desired_joint_speed, -1 * Eigen::VectorXd::Ones(6) * max_joint_velocity,
                            Eigen::VectorXd::Ones(6) * max_joint_velocity);

                            // std::cout << t().str() << "Relative time: " << time << std::endl;
                            // std::cout << t().str() << "Spline joint speed: " << spline_ref_speed << std::endl;
                            // std::cout << t().str() << "Desired joint speed: " << desired_joint_speed << std::endl;
                            // std::cout << t().str() << "Applied joint speed: " << clamped_joint_speed << std::endl;

                        control.speedJ(utils::eigen_to_std(clamped_joint_speed), joint_acceleration, 0.0001);

                        // control.zeroFtSensor();     // is good normally, but not with box handover
                        control.setWatchdog(5);

                        // update target pose
                        update_robot_state();

                        if (verbose) {
                            std::cout << t().str()
                                      << "Measured joint feedback (rad): " << curr_Q.transpose() << "\n";
                        }
                        applied_speed.position.setZero();
                        applied_speed.quat = Eigen::Quaterniond::Identity();
                        target_pose = current_pose.copy();
        
                        prev_mode = current_mode;
                    }
                    target_Q.resize(0);
                    continue;
                }
            } else if (current_mode == ControllerMode::EEPoseTarget) {
                applied_speed = calculate_speed(applied_speed);
                limiter.apply_position(applied_speed);
                if (calibration_guard.enabled) {
                    auto commanded = applied_speed.get_actual_speed_6D(deadspace_speed);
                    Eigen::Matrix<double,6,1> limited;
                    for(int i=0;i<6;++i) limited[i]=commanded[i];
                    limited=calibration_guard.limit_speed(limited, guard_previous_velocity, frequency);
                    guard_previous_velocity=limited;
                    success &= control.speedL(utils::eigen_to_std(limited), calibration_guard.value("max_linear_acceleration_m_s2"), 1./frequency);
                } else {
                    success &= control.speedL(applied_speed.get_actual_speed_6D(deadspace_speed),
                                              orientation_acceleration, 1. / frequency);
                }

                if (verbose && (applied_speed.position.array().abs() > deadspace_speed).any()) {
                    // Print applied speed, elapsed time, and pose error with orientation
                    std::cout << success << "  " << t().str();
                    std::cout << "pose: " << current_pose.str() << " ";
                    std::cout << "v: " << applied_speed.str();
                    const double pose_error_euler_scalar =
                        (target_pose.position - current_pose.position).norm();
                    const Eigen::Quaterniond pose_error_orientation =
                        target_pose.quat * current_pose.quat.inverse();
                    const double pose_error_orientation_scalar =
                        pose_error_orientation.angularDistance(Eigen::Quaterniond::Identity());
                    std::cout << std::fixed << std::setprecision(5)
                              << "  ep: " << pose_error_euler_scalar
                              << "  eo: " << pose_error_orientation_scalar << "\n";
                }
            } else if (current_mode == ControllerMode::ReferencePath) {
                const auto current_time = std::chrono::system_clock::now();
                const double time = (current_time - path_tracking_start_time).count() / 1e9;

                const Eigen::VectorXd spline_ref_speed = ref_path.get_velocity(time);
                const Eigen::VectorXd spline_ref_pos = ref_path.get_position(time);

                // TODO: config parameter
                const Eigen::VectorXd desired_joint_speed =
                    spline_ref_speed + path_tracking_p_gain * (spline_ref_pos - curr_Q);

                const Eigen::VectorXd clamped_joint_speed = utils::clamp(
                    desired_joint_speed, -1 * Eigen::VectorXd::Ones(6) * max_joint_velocity,
                    Eigen::VectorXd::Ones(6) * max_joint_velocity);

                    // std::cout << t().str() << "Relative time: " << time << std::endl;
                    // std::cout << t().str() << "Spline joint speed: " << spline_ref_speed << std::endl;
                    // std::cout << t().str() << "Desired joint speed: " << desired_joint_speed << std::endl;
                    // std::cout << t().str() << "Applied joint speed: " << clamped_joint_speed << std::endl;

                control.speedJ(utils::eigen_to_std(clamped_joint_speed), joint_acceleration, 0.01);
            }

            const auto next_cycle_time = start + std::chrono::duration<double>(1.0 / frequency);
            std::this_thread::sleep_until(next_cycle_time);
        }
        std::cout << t().str() << "UR5 Controller stopped gracefully\n";
    }

    void stop() {
        stop_flag.store(true);
        if (!calibration_guard.enabled) control.setWatchdog(0.001);  // preserve guarded watchdog during stop
        auto success = control.speedStop();
        if (success) {
            std::cout << t().str() << "UR5Controller speedStop applied\n";
        } else {
            std::cerr << t().str() << "UR5Controller speedStop failed\n";
        }
    }

   private:
    // Control and ZeroMQ
    zmq::context_t context;
    zmq::socket_t socket;
    zmq::socket_t publisher;
    
    CalibrationGuard calibration_guard;
    FeedbackClock rtde_clock;
    bool rtde_feedback_failed = false;
    Eigen::Matrix<double,6,1> guard_previous_velocity = Eigen::Matrix<double,6,1>::Zero();
    ur_rtde::RTDEControlInterface control;
    ur_rtde::RTDEReceiveInterface receive;
    ur_rtde::DashboardClient dashboard_client;
    
    std::atomic<bool> stop_flag;
    bool verbose;

    // config params
    std::vector<double> kpd_values;
    std::vector<double> reset_pose;
    double frequency;
    double p_delta;
    double o_delta;
    double exp_filter;
    double position_acceleration;
    double orientation_acceleration;
    double deadspace_speed;

    // trajectory tracking prameters
    double max_joint_velocity;
    double joint_acceleration;
    double path_tracking_p_gain;
    double joint_command_timeout_s;

    // internal state
    ControllerMode current_mode{ControllerMode::Idle};
    ControllerMode prev_mode{ControllerMode::EEPoseTarget};

    // indicates status of various ur5 commands
    bool success = true;
    bool reset_force_sensor = false;
    bool stop_requested = false;
    bool joint_command_stream_active = false;
    std::chrono::steady_clock::time_point last_joint_command_time{
        std::chrono::steady_clock::now()};

    Pose7D current_pose;
    Pose7D target_pose;
    TkSpline ref_path;
    std::chrono::time_point<std::chrono::high_resolution_clock> path_tracking_start_time;

    Eigen::VectorXd target_Q;
    Eigen::Vector<double, 6> curr_vel;
    Eigen::Vector<double, 6> curr_Q;
    Eigen::Vector<double, 6> curr_Qd;
    Eigen::Vector<double, 6> curr_force;
    Eigen::Vector<double, 6> curr_force_lowpass;

    // for speed calculation
    Pose7D pose_diff;
    Pose7D pose_vel_diff;
    Pose7D ee_speed;
    Pose7D applied_speed;
};

int main(int argc, char *argv[]) {
    const std::string config_path = (argc > 1) ? argv[1] : "config.json";

    json config;
    try {
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) throw std::runtime_error("Unable to open " + config_path);
        config_file >> config;
        if (config.contains("calibration_safety_path")) {
            std::ifstream safety_file(config.at("calibration_safety_path").get<std::string>());
            if (!safety_file.is_open()) throw std::runtime_error("Cannot open calibration safety file");
            safety_file >> config["calibration_safety"];
            CalibrationGuard validate(config["calibration_safety"]); // validate before connecting to robot
        }
        std::cout << "Loaded configuration from " << config_path << "\n";
    } catch (const std::exception &e) {
        std::cerr << "Error loading controller configuration: " << e.what() << "\n";
        return 1;
    }

#ifdef CALIBRATION_ONLY
    if (!config.contains("calibration_safety") || !config.at("calibration_safety").is_object()) {
        std::cerr << "The calibration-only controller requires verified calibration_safety settings.\n";
        return 1;
    }
#endif
    std::cout << "Setting up controller" << std::endl;
    UR5Controller controller(config["robot_ip"].get<std::string>(), config);

    std::cout << "Running" << std::endl;
    static UR5Controller *c_ptr = &controller;
    try {
        // Handle Ctrl+C signal
        static std::atomic_flag signal_handled = ATOMIC_FLAG_INIT;
        std::signal(SIGINT, [](int) {
            if (!signal_handled.test_and_set()) {
                std::cout << "\nCtrl+C detected. Stopping the controller...\n";
                if (c_ptr != nullptr) {
                    c_ptr->stop();
                }
            }
        });
        controller.run();
    } catch (const std::exception &e) {
        if (c_ptr != nullptr) {
            c_ptr->stop();
        }
        std::cerr << e.what() << '\n';
    }
    return 0;
}
