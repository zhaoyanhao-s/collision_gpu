#pragma once
struct NumericalIntegration
{
	template <typename F>
	static double
		simpson_3_8(F&& derivative_func, const float& L, const float& R)
	{
		double mid_L = (2 * L + R) / 3.0, mid_R = (L + 2 * R) / 3.0;
		return (derivative_func(L) +
			3.0 * derivative_func(mid_L) +
			3.0 * derivative_func(mid_R) +
			derivative_func(R)) * (R - L) / 8.0;
	}

	template <typename F>
	static double
		adaptive_simpson_3_8(F&& derivative_func,
            const float& L, const float& R, const float& eps = 0.001)
	{
		const float mid = (L + R) / 2.0;
		float ST = simpson_3_8(derivative_func, L, R),
			SL = simpson_3_8(derivative_func, L, mid),
			SR = simpson_3_8(derivative_func, mid, R);
		float ans = SL + SR - ST;

        if (fabs(ans) <= 15.0 * eps)  return SL + SR + ans / 15.0;
		return adaptive_simpson_3_8(derivative_func, L, mid, eps / 2.0) +
			adaptive_simpson_3_8(derivative_func, mid, R, eps / 2.0);
	}
};
