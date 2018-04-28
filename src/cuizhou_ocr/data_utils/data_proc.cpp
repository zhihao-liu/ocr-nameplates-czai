//
// Created by Zhihao Liu on 4/26/18.
//

#include "data_utils/data_proc.hpp"
#include <cassert>


namespace cz {

LinearFit::LinearFit(std::vector<double> const& x, std::vector<double> const& y) {
    assert(x.size() == y.size());

    double t1 = 0, t2 = 0, t3 = 0, t4 = 0;
    for (int i = 0; i < x.size(); ++i) {
        t1 += x[i] * x[i];
        t2 += x[i];
        t3 += x[i] * y[i];
        t4 += y[i];
    }

    a = (t3 * x.size() - t2 * t4) / (t1 * x.size() - t2 * t2);
    b = (t1 * t4 - t2 * t3) / (t1 * x.size() - t2 * t2);
}

double LinearFit::slope() const {
    return a;
}

double LinearFit::constant() const {
    return b;
}

} // end namespace cz
