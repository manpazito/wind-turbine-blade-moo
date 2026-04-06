#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

constexpr double kPi = 3.14159265358979323846;

double clamp(double value, double lo, double hi) {
    return std::min(std::max(value, lo), hi);
}

double prandtl_loss_factor(
    int blades,
    double radius_m,
    double hub_radius_m,
    double r_m,
    double phi_rad
) {
    double sin_phi = std::abs(std::sin(phi_rad));
    sin_phi = std::max(sin_phi, 1e-6);
    const double denom = std::max(r_m * sin_phi, 1e-6);
    const double tip_exp = -0.5 * static_cast<double>(blades) * (radius_m - r_m) / denom;
    const double root_exp = -0.5 * static_cast<double>(blades) * (r_m - hub_radius_m) / denom;
    const double tip_term = std::exp(clamp(tip_exp, -60.0, 0.0));
    const double root_term = std::exp(clamp(root_exp, -60.0, 0.0));
    const double tip_loss = (2.0 / kPi) * std::acos(clamp(tip_term, 0.0, 1.0));
    const double root_loss = (2.0 / kPi) * std::acos(clamp(root_term, 0.0, 1.0));
    return std::max(tip_loss * root_loss, 1e-3);
}

double interp_1d(
    const double* x,
    const double* y,
    ssize_t n,
    double query
) {
    if (n <= 0) {
        return 0.0;
    }
    if (query <= x[0]) {
        return y[0];
    }
    if (query >= x[n - 1]) {
        return y[n - 1];
    }

    ssize_t lo = 0;
    ssize_t hi = n - 1;
    while (hi - lo > 1) {
        const ssize_t mid = lo + (hi - lo) / 2;
        if (x[mid] <= query) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    const double x0 = x[lo];
    const double x1 = x[hi];
    const double y0 = y[lo];
    const double y1 = y[hi];
    const double t = (query - x0) / std::max(x1 - x0, 1e-12);
    return y0 + t * (y1 - y0);
}

void sample_cl_cd(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& re_bins,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& alpha_grid,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& cl_grid,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& cd_grid,
    double reynolds,
    double alpha_deg,
    double* out_cl,
    double* out_cd
) {
    const auto re = re_bins.unchecked<1>();
    const auto alpha = alpha_grid.unchecked<1>();
    const auto cl = cl_grid.unchecked<2>();
    const auto cd = cd_grid.unchecked<2>();

    const ssize_t n_re = re.shape(0);
    const ssize_t n_alpha = alpha.shape(0);
    const double* alpha_ptr = alpha.data(0);

    auto row_interp = [&](ssize_t re_idx, bool do_cl) {
        const double* y_ptr = do_cl ? cl.data(re_idx, 0) : cd.data(re_idx, 0);
        return interp_1d(alpha_ptr, y_ptr, n_alpha, alpha_deg);
    };

    if (reynolds <= re(0)) {
        *out_cl = row_interp(0, true);
        *out_cd = std::max(row_interp(0, false), 1e-5);
        return;
    }
    if (reynolds >= re(n_re - 1)) {
        *out_cl = row_interp(n_re - 1, true);
        *out_cd = std::max(row_interp(n_re - 1, false), 1e-5);
        return;
    }

    ssize_t lo = 0;
    ssize_t hi = n_re - 1;
    while (hi - lo > 1) {
        const ssize_t mid = lo + (hi - lo) / 2;
        if (re(mid) <= reynolds) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    const double cl_lo = row_interp(lo, true);
    const double cl_hi = row_interp(hi, true);
    const double cd_lo = row_interp(lo, false);
    const double cd_hi = row_interp(hi, false);

    const double w = (reynolds - re(lo)) / std::max(re(hi) - re(lo), 1e-12);
    *out_cl = (1.0 - w) * cl_lo + w * cl_hi;
    *out_cd = std::max((1.0 - w) * cd_lo + w * cd_hi, 1e-5);
}

}  // namespace

py::dict evaluate_rotor_cpp(
    double radius_m,
    double wind_speed_ms,
    double air_density,
    double dynamic_viscosity,
    double pitch_deg,
    int blades,
    double tip_speed_ratio,
    double hub_radius_ratio,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& r_m,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& chord_m,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& twist_deg,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& re_bins,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& alpha_grid,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& cl_grid,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& cd_grid
) {
    if (r_m.ndim() != 1 || chord_m.ndim() != 1 || twist_deg.ndim() != 1) {
        throw std::runtime_error("r_m, chord_m, and twist_deg must be 1-D arrays.");
    }
    if (re_bins.ndim() != 1 || alpha_grid.ndim() != 1) {
        throw std::runtime_error("re_bins and alpha_grid must be 1-D arrays.");
    }
    if (cl_grid.ndim() != 2 || cd_grid.ndim() != 2) {
        throw std::runtime_error("cl_grid and cd_grid must be 2-D arrays.");
    }

    const ssize_t n_sections = r_m.shape(0);
    if (chord_m.shape(0) != n_sections || twist_deg.shape(0) != n_sections) {
        throw std::runtime_error("Section arrays must have matching lengths.");
    }

    const ssize_t n_re = re_bins.shape(0);
    const ssize_t n_alpha = alpha_grid.shape(0);
    if (cl_grid.shape(0) != n_re || cl_grid.shape(1) != n_alpha) {
        throw std::runtime_error("cl_grid shape must match (len(re_bins), len(alpha_grid)).");
    }
    if (cd_grid.shape(0) != n_re || cd_grid.shape(1) != n_alpha) {
        throw std::runtime_error("cd_grid shape must match (len(re_bins), len(alpha_grid)).");
    }

    const auto r = r_m.unchecked<1>();
    const auto c = chord_m.unchecked<1>();
    const auto twist = twist_deg.unchecked<1>();

    const double hub_radius_m = hub_radius_ratio * radius_m;
    const double omega = tip_speed_ratio * wind_speed_ms / radius_m;
    const double dr = (radius_m - hub_radius_m) / std::max<ssize_t>(n_sections, 1);

    std::vector<double> phi_deg_v;
    std::vector<double> alpha_deg_v;
    std::vector<double> reynolds_v;
    std::vector<double> cl_v;
    std::vector<double> cd_v;
    std::vector<double> a_v;
    std::vector<double> a_prime_v;
    std::vector<double> cn_v;
    std::vector<double> ct_v;
    std::vector<double> dthrust_v;
    std::vector<double> dtorque_v;
    std::vector<double> solidity_v;

    phi_deg_v.reserve(static_cast<size_t>(n_sections));
    alpha_deg_v.reserve(static_cast<size_t>(n_sections));
    reynolds_v.reserve(static_cast<size_t>(n_sections));
    cl_v.reserve(static_cast<size_t>(n_sections));
    cd_v.reserve(static_cast<size_t>(n_sections));
    a_v.reserve(static_cast<size_t>(n_sections));
    a_prime_v.reserve(static_cast<size_t>(n_sections));
    cn_v.reserve(static_cast<size_t>(n_sections));
    ct_v.reserve(static_cast<size_t>(n_sections));
    dthrust_v.reserve(static_cast<size_t>(n_sections));
    dtorque_v.reserve(static_cast<size_t>(n_sections));
    solidity_v.reserve(static_cast<size_t>(n_sections));

    double thrust_n = 0.0;
    double torque_nm = 0.0;
    double root_moment_nm = 0.0;

    for (ssize_t idx = 0; idx < n_sections; ++idx) {
        double a = 0.30;
        double a_prime = 0.00;

        bool has_last = false;
        double last_phi_deg = 0.0;
        double last_alpha_deg = 0.0;
        double last_reynolds = 0.0;
        double last_cl = 0.0;
        double last_cd = 0.0;
        double last_a = 0.0;
        double last_a_prime = 0.0;
        double last_cn = 0.0;
        double last_ct = 0.0;
        double last_dthrust = 0.0;
        double last_dtorque = 0.0;
        double last_sigma = 0.0;

        for (int iter = 0; iter < 120; ++iter) {
            const double v_axial = wind_speed_ms * (1.0 - a);
            const double v_tan = omega * r(idx) * (1.0 + a_prime);
            const double phi = std::atan2(std::max(v_axial, 1e-8), std::max(v_tan, 1e-8));
            const double w_rel = std::hypot(v_axial, v_tan);
            const double alpha_deg_local = (phi * 180.0 / kPi) - (twist(idx) + pitch_deg);
            const double reynolds = air_density * w_rel * c(idx) / dynamic_viscosity;

            double cl = 0.0;
            double cd = 0.0;
            sample_cl_cd(re_bins, alpha_grid, cl_grid, cd_grid, reynolds, alpha_deg_local, &cl, &cd);

            const double cn = cl * std::cos(phi) + cd * std::sin(phi);
            const double ct = cl * std::sin(phi) - cd * std::cos(phi);
            const double sigma = static_cast<double>(blades) * c(idx) / (2.0 * kPi * r(idx));
            const double f_loss = prandtl_loss_factor(blades, radius_m, hub_radius_m, r(idx), phi);

            const double sin_phi = std::max(std::abs(std::sin(phi)), 1e-5);
            const double cos_phi = std::max(std::abs(std::cos(phi)), 1e-5);
            const double denom_a = (4.0 * f_loss * sin_phi * sin_phi) / std::max(sigma * cn, 1e-8);
            const double denom_ap = (4.0 * f_loss * sin_phi * cos_phi) / std::max(sigma * ct, 1e-8);

            double a_new = 1.0 / (denom_a + 1.0);
            double a_prime_new = 1.0 / (denom_ap - 1.0);
            a_new = clamp(a_new, 0.0, 0.95);
            a_prime_new = clamp(a_prime_new, -0.5, 0.5);

            const double a_upd = 0.75 * a + 0.25 * a_new;
            const double ap_upd = 0.75 * a_prime + 0.25 * a_prime_new;

            const double dyn_pressure = 0.5 * air_density * w_rel * w_rel;
            const double d_lift = dyn_pressure * c(idx) * cl * dr;
            const double d_drag = dyn_pressure * c(idx) * cd * dr;
            const double dthrust = static_cast<double>(blades) * (d_lift * std::cos(phi) + d_drag * std::sin(phi));
            const double dtorque = static_cast<double>(blades) * (d_lift * std::sin(phi) - d_drag * std::cos(phi)) * r(idx);

            has_last = true;
            last_phi_deg = phi * 180.0 / kPi;
            last_alpha_deg = alpha_deg_local;
            last_reynolds = reynolds;
            last_cl = cl;
            last_cd = cd;
            last_a = a_upd;
            last_a_prime = ap_upd;
            last_cn = cn;
            last_ct = ct;
            last_dthrust = dthrust;
            last_dtorque = dtorque;
            last_sigma = sigma;

            if (std::abs(a_upd - a) < 1e-4 && std::abs(ap_upd - a_prime) < 1e-4) {
                a = a_upd;
                a_prime = ap_upd;
                break;
            }
            a = a_upd;
            a_prime = ap_upd;
        }

        if (!has_last) {
            continue;
        }

        thrust_n += last_dthrust;
        torque_nm += last_dtorque;
        root_moment_nm += last_dthrust * std::max(r(idx) - hub_radius_m, 0.0);

        phi_deg_v.push_back(last_phi_deg);
        alpha_deg_v.push_back(last_alpha_deg);
        reynolds_v.push_back(last_reynolds);
        cl_v.push_back(last_cl);
        cd_v.push_back(last_cd);
        a_v.push_back(last_a);
        a_prime_v.push_back(last_a_prime);
        cn_v.push_back(last_cn);
        ct_v.push_back(last_ct);
        dthrust_v.push_back(last_dthrust);
        dtorque_v.push_back(last_dtorque);
        solidity_v.push_back(last_sigma);
    }

    const double swept_area = kPi * radius_m * radius_m;
    const double power_w = omega * torque_nm;
    const double cp = power_w / (0.5 * air_density * swept_area * std::pow(wind_speed_ms, 3) + 1e-9);
    const double ct = thrust_n / (0.5 * air_density * swept_area * std::pow(wind_speed_ms, 2) + 1e-9);

    double solidity_mean = 0.0;
    if (!solidity_v.empty()) {
        for (double v : solidity_v) {
            solidity_mean += v;
        }
        solidity_mean /= static_cast<double>(solidity_v.size());
    }

    py::dict out;
    out["phi_deg"] = py::array_t<double>(phi_deg_v.size(), phi_deg_v.data());
    out["alpha_deg"] = py::array_t<double>(alpha_deg_v.size(), alpha_deg_v.data());
    out["reynolds"] = py::array_t<double>(reynolds_v.size(), reynolds_v.data());
    out["cl"] = py::array_t<double>(cl_v.size(), cl_v.data());
    out["cd"] = py::array_t<double>(cd_v.size(), cd_v.data());
    out["a"] = py::array_t<double>(a_v.size(), a_v.data());
    out["a_prime"] = py::array_t<double>(a_prime_v.size(), a_prime_v.data());
    out["cn"] = py::array_t<double>(cn_v.size(), cn_v.data());
    out["ct_section"] = py::array_t<double>(ct_v.size(), ct_v.data());
    out["dthrust_n"] = py::array_t<double>(dthrust_v.size(), dthrust_v.data());
    out["dtorque_nm"] = py::array_t<double>(dtorque_v.size(), dtorque_v.data());
    out["local_solidity"] = py::array_t<double>(solidity_v.size(), solidity_v.data());

    out["cp"] = cp;
    out["ct"] = ct;
    out["power_w"] = power_w;
    out["thrust_n"] = thrust_n;
    out["torque_nm"] = torque_nm;
    out["root_moment_nm"] = root_moment_nm;
    out["solidity_mean"] = solidity_mean;

    return out;
}

PYBIND11_MODULE(_bem_cpp, m) {
    m.doc() = "C++-accelerated BEM rotor evaluation";

    m.def(
        "evaluate_rotor_cpp",
        &evaluate_rotor_cpp,
        py::arg("radius_m"),
        py::arg("wind_speed_ms"),
        py::arg("air_density"),
        py::arg("dynamic_viscosity"),
        py::arg("pitch_deg"),
        py::arg("blades"),
        py::arg("tip_speed_ratio"),
        py::arg("hub_radius_ratio"),
        py::arg("r_m"),
        py::arg("chord_m"),
        py::arg("twist_deg"),
        py::arg("re_bins"),
        py::arg("alpha_grid"),
        py::arg("cl_grid"),
        py::arg("cd_grid")
    );
}
