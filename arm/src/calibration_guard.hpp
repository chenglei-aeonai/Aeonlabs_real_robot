#pragma once
// Application-level guard for calibration. Independent UR safety setup remains required.
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "collision_plane_model.hpp"

class CalibrationGuard {
 public:
    using Json = nlohmann::json;
    using Clock = std::chrono::steady_clock;
    bool enabled = false, armed = false, moving = false, plane_only = false;
    CollisionPlaneModel surface_model;
    std::vector<double> hand_q, initial_hand_q;
    Clock::time_point hand_valid_until;
    std::string fault, session;
    Json config, measured_speed;
    std::string speed_warning;
    Eigen::Vector3d lower, upper, plane_normal;
    double plane_offset = 0;
    double radius = 0, reserve = 0;
    std::vector<double> anchor, anchor_q;
    Clock::time_point last_command;

    explicit CalibrationGuard(const Json &settings = nullptr) : config(settings) {
        if (settings.is_null()) return;
        enabled = true;
        const auto mode=config.value("geometry_mode",std::string("workspace"));
        require(mode=="workspace"||mode=="plane_only","Unknown geometry mode");
        plane_only=mode=="plane_only";
        if(plane_only)require(config.value("allow_estimated_plane",false),"Plane-only estimate must be explicitly acknowledged");
        if(!plane_only)require(config.value("verified", false) && config.value("robot_safety_setup_verified", false), "Unverified calibration safety configuration");
        if(!plane_only)require(!config.at("verification_notes").get<std::string>().empty(), "Missing verification notes");
        if(!plane_only)for (const auto &entry : std::vector<std::pair<std::string, int>>{
                 {"workspace_min_m",3}, {"workspace_max_m",3}, {"table_min_xy_m",2},
                 {"table_max_xy_m",2}, {"arm_link_radii_m",6}}) {
            const auto &v = config.at(entry.first);
            require(v.is_array() && v.size() == static_cast<size_t>(entry.second), "Invalid geometry vector");
            for (const auto &x : v) require(x.is_number() && std::isfinite(x.get<double>()), "Nonfinite geometry");
        }
        for (const std::string &key : {"table_top_z_m", "tool_radius_m", "clearance_m", "uncertainty_m", "stopping_margin_m",
             "max_linear_speed_m_s", "max_angular_speed_deg_s", "max_linear_acceleration_m_s2", "max_angular_acceleration_deg_s2",
             "command_timeout_s", "state_timeout_s", "motion_timeout_s", "max_joint_deviation_deg", "max_joint_speed_deg_s",
             "max_start_translation_m", "tilt_deg", "position_tolerance_m", "angle_tolerance_deg", "settle_s", "max_reprojection_error_px"}) {
            if(plane_only&&(key=="table_top_z_m"||key=="tool_radius_m"))continue;
            const auto &v = config.at(key);
            require(v.is_number() && std::isfinite(v.get<double>()) && (key == "table_top_z_m" || v.get<double>() > 0), "Invalid safety scalar: " + key);
        }
        for (const auto &cap : std::vector<std::pair<std::string,double>>{
                 {"max_linear_speed_m_s",.01}, {"max_angular_speed_deg_s",2},
                 {"max_linear_acceleration_m_s2",.02}, {"max_angular_acceleration_deg_s2",4},
                 {"command_timeout_s",.2}, {"state_timeout_s",.2}, {"max_start_translation_m",.08},
                 {"tilt_deg",3}, {"max_joint_deviation_deg",20}, {"max_joint_speed_deg_s",5},
                 {"position_tolerance_m",.002}, {"angle_tolerance_deg",.3}})
            require(value(cap.first) <= cap.second, "Calibration cap exceeded: " + cap.first);
        for(const auto &entry:std::vector<std::pair<std::string,double>>{{"measured_tcp_speed_trip_factor",2.},{"measured_joint_speed_trip_factor",1.5}}){
            const auto factor=config.value(entry.first,Json(entry.second));
            require(factor.is_number()&&std::isfinite(factor.get<double>())&&factor.get<double>()>1.25&&factor.get<double>()<=entry.second,"Invalid measured-speed trip factor");
        }
        require(value("command_timeout_s") >= .08 && value("state_timeout_s") >= .08, "Timeout below minimum");
        require(value("clearance_m") >= .02 && value("uncertainty_m") >= .01 && value("stopping_margin_m") >= .01, "Insufficient reserves");
        require(value("settle_s")>=.5,"Insufficient settling");
        if(!plane_only)require(value("tool_radius_m") >= .1,"Insufficient tool radius");
        if(!plane_only)for (double r : config.at("arm_link_radii_m").get<std::vector<double>>()) require(r >= .05, "Arm radius too small");
        require(config.at("samples_per_pose").is_number_integer() && config.at("samples_per_pose").get<int>() >= 3 && config.at("samples_per_pose").get<int>() <= 30, "Invalid burst count");
        const auto &plane = config.at("table_plane");
        require(plane.is_object() && plane.at("frame").get<std::string>() == "robot_base", "Table plane must be in fixed robot_base frame");
        const auto &normal = plane.at("normal");
        require(normal.is_array() && normal.size() == 3, "Invalid table-plane normal");
        for (const auto &v : normal) require(v.is_number() && std::isfinite(v.get<double>()), "Nonfinite plane normal");
        require(plane.at("offset_m").is_number() && std::isfinite(plane.at("offset_m").get<double>()), "Nonfinite plane offset");
        plane_normal=vec(normal); plane_offset=plane.at("offset_m").get<double>();
        require(std::abs(plane_normal.norm()-1.) <= 1e-6 && plane_normal.z() > 1e-6, "Plane normal must be unit and upward");
        if(!plane_only)for(double x : {config.at("table_min_xy_m")[0].get<double>(),config.at("table_max_xy_m")[0].get<double>()})
            for(double y : {config.at("table_min_xy_m")[1].get<double>(),config.at("table_max_xy_m")[1].get<double>()})
                require(-(plane_normal.x()*x+plane_normal.y()*y+plane_offset)/plane_normal.z() <= value("table_top_z_m")+1e-9,
                        "Table height bound must enclose the full plane and fixtures");
        const auto down=config.value("max_downward_offset_m",Json(0.));
        require(down.is_number() && std::isfinite(down.get<double>()) && down.get<double>() >= 0 && down.get<double>() <= .05,
                "Downward calibration offset must be between 0 and 0.05m");
        reserve = value("uncertainty_m") + value("stopping_margin_m");
        radius = (plane_only ? 0. : value("tool_radius_m")) + reserve;
        if(plane_only){
            surface_model.load(config.at("collision_model_path").get<std::string>());
            return;
        }
        lower = vec(config.at("workspace_min_m")); upper = vec(config.at("workspace_max_m"));
        require(((lower.array() + radius) < (upper.array() - radius)).all(), "Workspace cannot contain tool envelope");
        for (int i = 0; i < 2; ++i) {
            require(config.at("table_min_xy_m")[i].get<double>() < config.at("table_max_xy_m")[i].get<double>(), "Invalid table bounds");
            require(lower[i] >= config.at("table_min_xy_m")[i].get<double>() && upper[i] <= config.at("table_max_xy_m")[i].get<double>(), "Workspace XY must be inside table footprint");
        }
    }

    double value(const std::string &key) const { return config.at(key).get<double>(); }
    static void require(bool condition, const std::string &message) { if (!condition) throw std::runtime_error(message); }
    static Eigen::Vector3d vec(const Json &j) { return {j[0].get<double>(), j[1].get<double>(), j[2].get<double>()}; }
    static Eigen::Vector3d position(const std::vector<double> &p) { return {p[0],p[1],p[2]}; }
    static Eigen::Quaterniond rotation(const std::vector<double> &p) { return {p[6],p[3],p[4],p[5]}; }
    static double radians(double deg) { return deg * 3.14159265358979323846 / 180.; }

    void check_pose(const std::vector<double> &p, bool check_anchor = true, bool measured = false) const {
        require(p.size() == 7, "Invalid TCP pose size");
        for (double x : p) require(std::isfinite(x), "Nonfinite TCP pose");
        require(std::abs(rotation(p).norm() - 1) <= .001, "Non-unit TCP quaternion");
        auto xyz = position(p);
        require(plane_normal.dot(xyz)+plane_offset-radius >= value("clearance_m"), "Full hand/board envelope too close to fixed table plane");
        if(!plane_only){
            require(((xyz.array() - radius) >= lower.array()).all() && ((xyz.array() + radius) <= upper.array()).all(), "Tool envelope leaves workspace");
            require(xyz.z() - radius >= value("table_top_z_m") + value("clearance_m"), "Hand/board too close to table");
        }
        if (check_anchor && !anchor.empty()) {
            require((xyz - position(anchor)).norm() <= value("max_start_translation_m"), "TCP leaves local neighbourhood");
            require(xyz.z() >= anchor[2] - config.value("max_downward_offset_m",0.) - (measured ? value("position_tolerance_m") : 1e-9), "TCP below approved starting-height offset");
            require(rotation(p).angularDistance(rotation(anchor)) <= radians(value("tilt_deg") + (measured ? value("angle_tolerance_deg") : 0.)) + 1e-8, "Absolute tilt exceeds limit");
        }
    }

    void check_arm(const std::vector<double> &q) const {
        require(q.size() == 6, "Invalid arm joints");
        for (int i = 0; i < 6; ++i) {
            require(std::isfinite(q[i]), "Nonfinite arm joint");
            require(std::abs(q[i]) <= radians(i == 2 ? 180 : 360), "Arm exceeds conservative URDF joint limit");
            if (!anchor_q.empty()) require(std::abs(q[i] - anchor_q[i]) <= radians(value("max_joint_deviation_deg")), "Arm leaves joint neighbourhood");
        }
        if(plane_only)return; // Actual moving URDF surfaces are checked in tick().
        const double a[6] = {0,-.425,-.3922,0,0,0};
        const double d[6] = {.1625,0,0,.1333,.0997,.0996};
        const double alpha[6] = {radians(90),0,0,radians(90),radians(-90),0};
        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        Eigen::Vector3d start = Eigen::Vector3d::Zero();
        for (int i = 0; i < 6; ++i) {
            const double c=std::cos(q[i]), s=std::sin(q[i]), ca=std::cos(alpha[i]), sa=std::sin(alpha[i]);
            Eigen::Matrix4d dh;
            dh << c,-s*ca,s*sa,a[i]*c, s,c*ca,-c*sa,a[i]*s, 0,sa,ca,d[i], 0,0,0,1;
            transform *= dh;
            Eigen::Vector3d end = transform.block<3,1>(0,3);
            const double r = config.at("arm_link_radii_m")[i].get<double>() + reserve + value("clearance_m");
            Eigen::Vector3d lo(config.at("table_min_xy_m")[0].get<double>()-r, config.at("table_min_xy_m")[1].get<double>()-r, -1e6);
            Eigen::Vector3d hi(config.at("table_max_xy_m")[0].get<double>()+r, config.at("table_max_xy_m")[1].get<double>()+r, value("table_top_z_m")+r);
            double enter=0, leave=1;
            for (int axis = 0; axis < 3; ++axis) {
                const double delta = end[axis] - start[axis];
                if (std::abs(delta) < 1e-12) {
                    if (start[axis] < lo[axis] || start[axis] > hi[axis]) { enter=1;leave=0;break; }
                } else {
                    double first=(lo[axis]-start[axis])/delta, last=(hi[axis]-start[axis])/delta;
                    if (first > last) std::swap(first,last);
                    enter=std::max(enter,first); leave=std::min(leave,last);
                }
            }
            require(enter > leave, "Arm capsule approaches table: link " + std::to_string(i+1));
            start=end;
        }
    }

    void arm(const std::string &id, const std::vector<double> &p, const std::vector<double> &q) {
        require(fault.empty(), "Guard fault is latched; operator inspection/restart required");
        require(!id.empty(), "Missing calibration session");
        if (armed) { require(session == id, "Different session already armed"); return; }
        check_pose(p, false); check_arm(q);
        if(plane_only){
            require(Clock::now()<=hand_valid_until,"Fresh hand feedback required before arming");
            surface_model.check(q,hand_q,p,plane_normal,plane_offset,reserve+value("clearance_m"));
            initial_hand_q=hand_q;
        }
        anchor=p; anchor_q=q; session=id; armed=true; moving=false;
    }

    void target(const std::string &id, const std::vector<double> &p, Clock::time_point now) {
        require(armed && fault.empty() && session == id, "Invalid/unarmed calibration session");
        check_pose(p);
        moving=true; last_command=now;
    }

    void tick(const std::vector<double> &p, const std::vector<double> &q,
              const std::vector<double> &velocity, const std::vector<double> &qd, Clock::time_point now) {
        check_pose(p, true, true); check_arm(q);
        if(plane_only&&!hand_q.empty()){
            if(moving)require(now<=hand_valid_until,"Hand feedback stale during motion");
            surface_model.check(q,hand_q,p,plane_normal,plane_offset,reserve+value("clearance_m"));
        }
        require(velocity.size() == 6 && qd.size() == 6, "Missing velocity feedback");
        double linear2=0, angular2=0, joint_max=0;
        for (int i=0;i<6;++i) {
            require(std::isfinite(velocity[i]) && std::isfinite(qd[i]), "Nonfinite velocity feedback");
            joint_max=std::max(joint_max,std::abs(qd[i]));
            if (i<3) linear2+=velocity[i]*velocity[i]; else angular2+=velocity[i]*velocity[i];
        }
        const double linear=std::sqrt(linear2), angular_deg=std::sqrt(angular2)/radians(1), joint_deg=joint_max/radians(1);
        measured_speed={{"linear_m_s",linear},{"angular_deg_s",angular_deg},{"max_joint_deg_s",joint_deg}};
        const double tcp_factor=config.value("measured_tcp_speed_trip_factor",2.);
        const double joint_factor=config.value("measured_joint_speed_trip_factor",1.5);
        speed_warning=(linear>value("max_linear_speed_m_s")*1.25||angular_deg>value("max_angular_speed_deg_s")*1.25||joint_deg>value("max_joint_speed_deg_s")*1.25) ?
            "Measured speed above nominal tolerance; minor excursion, not a latched fault" : "";
        require(linear<=value("max_linear_speed_m_s")*tcp_factor&&angular_deg<=value("max_angular_speed_deg_s")*tcp_factor,
            "Severe measured TCP overspeed: "+std::to_string(linear*1000)+" mm/s, "+std::to_string(angular_deg)+" deg/s; thresholds "+std::to_string(value("max_linear_speed_m_s")*tcp_factor*1000)+" mm/s, "+std::to_string(value("max_angular_speed_deg_s")*tcp_factor)+" deg/s");
        require(joint_deg<=value("max_joint_speed_deg_s")*joint_factor,
            "Severe measured joint overspeed: "+std::to_string(joint_deg)+" deg/s; threshold "+std::to_string(value("max_joint_speed_deg_s")*joint_factor)+" deg/s");
        if (moving) require(std::chrono::duration<double>(now-last_command).count() <= value("command_timeout_s"), "Calibration command stream timed out");
    }

    Eigen::Matrix<double,6,1> limit_speed(const Eigen::Matrix<double,6,1> &command,
                                           const Eigen::Matrix<double,6,1> &previous, double frequency) const {
        require(command.allFinite() && previous.allFinite() && std::isfinite(frequency) && frequency > 0, "Invalid velocity command/frequency");
        Eigen::Matrix<double,6,1> result;
        for(int block=0;block<2;++block) {
            const double max_speed = block == 0 ? value("max_linear_speed_m_s") : radians(value("max_angular_speed_deg_s"));
            const double max_accel = block == 0 ? value("max_linear_acceleration_m_s2") : radians(value("max_angular_acceleration_deg_s2"));
            Eigen::Vector3d v=command.segment<3>(block*3);
            if(v.norm()>max_speed) v*=max_speed/v.norm();
            Eigen::Vector3d delta=v-previous.segment<3>(block*3);
            if(delta.norm()>max_accel/frequency) delta*=max_accel/frequency/delta.norm();
            result.segment<3>(block*3)=previous.segment<3>(block*3)+delta;
        }
        return result;
    }

    void update_hand(const Json &packet){
        if(!plane_only)return;
        auto positions=packet.at("hand_q").get<std::vector<double>>();
        require(positions.size()==20,"Need twenty measured hand joints");
        for(size_t i=0;i<positions.size();++i){
            require(std::isfinite(positions[i]),"Nonfinite hand feedback");
            if(!initial_hand_q.empty())require(std::abs(positions[i]-initial_hand_q[i])<=radians(2),"Hand posture changed during plane-only calibration");
        }
        const double now_ms=std::chrono::duration<double,std::milli>(std::chrono::system_clock::now().time_since_epoch()).count();
        const double age=(now_ms-packet.at("hand_timestamp_ms").get<double>())/1000.;
        require(std::isfinite(age)&&age>=-.05&&age<=.25,"Stale/invalid hand feedback timestamp");
        hand_q=positions;
        hand_valid_until=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(.25-std::max(0.,age)));
    }

    void trip(const std::string &reason) { if(fault.empty()) fault=reason; moving=false; armed=false; }
    Json status() const {
        if (!enabled) return nullptr;
        return {{"version",5},{"config",config},{"armed",armed},{"session",session},{"fault",fault},{"speed_warning",speed_warning},{"measured_speed",measured_speed}};
    }
};
