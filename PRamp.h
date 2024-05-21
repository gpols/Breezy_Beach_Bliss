#ifndef INC_PRAMP_H
#define INC_PRAMP_H

#include <cmath> // fmod, abs, cos, sin

// Periodic ramp with unit phase

// The output can be used directly to animate cyclic values 
// (rotation angles, hue, etc.). Several mapping functions
// are provided so the ramp can be used as a more generic
// oscillator.
// 
// Lance Putnam, 2022
class PRamp{
public:

	PRamp(float f=1., float p=0.): mFreq(f), mPhase(p){}

	// Set frequency, in Hertz
	PRamp& freq(float v) { mFreq = v; return *this; }
	// Get frequency, in Hertz
	float freq() const { return mFreq; }

	// Set period, in seconds
	PRamp& period(float v) { return freq(1./v); }

	// Set phase, in [0,1)
	PRamp& phase(float v){ mPhase=v; return *this; }
	// Get phase, in [0,1)
	float phase() const { return mPhase; }
	// Get phase, in [0, 2pi)
	float phaseRad() const { return mPhase * twoPi; }

	// Reset phase
	PRamp& reset(){ return phase(0.); }

	// Update phase
	PRamp& update(float deltaSec = 1.f) {
		mPhase = std::fmod(mPhase + deltaSec * mFreq, 1.f);
		return *this;
	}

	// Get triangle wave output, in [0,1]
	float tri() const {
		// https://www.desmos.com/calculator/ojpjy7ib0w
		return 1. - std::abs(2.*mPhase-1.);
	}
    // Map current phase to parabolic wave, in [0,1]
    float para() const {
        auto saw = 2.*mPhase-1.;
        return 1. - saw*saw;
    }

	// Get skewed triangle wave output, in [0,1]
	float tri(float skew) const {
		// https://www.desmos.com/calculator/jjsciw6z98
		constexpr float eps0 = 1e-8;
		constexpr float eps1 = 1. - eps0;
		skew = skew<eps0 ? eps0 : skew>eps1 ? eps1 : skew;
		if (mPhase < skew) return mPhase / skew;
		else return (1. - mPhase) / (1. - skew);
	}

	// Get sine wave; kth harmonic
	float sin(int k=1) const {
		return std::sin(k * phaseRad());
	}

	// Get haversine wave; kth harmonic
	float haversin(int k=1) const {
		return 0.5 - 0.5*this->sin(k);
	}

	// Get point on unit circle
	template <class Vec2>
	Vec2 circle(int k=1) const {
		auto t = k*phaseRad();
		return { std::cos(t), std::sin(t) };
	}

	// Get unit Archimedean spiral with equal sampling per arc length
	template <class Vec2>
	Vec2 spiral(float coils) const {
		auto r = std::sqrt(phase()); // use sqrt to get equal sampling w.r.t. arc length
		auto t = coils * r * twoPi;
		return { r * std::cos(t), r * std::sin(t) };
	}

	template <class Vec2>
	Vec2 bicircloid(int f1, int f2, float A1, float A2) const {
		return circle<Vec2>(f1)*A1 + circle<Vec2>(f2)*A2;
	}

	template <class Vec2>
	Vec2 rose(int f1, int f2) const {
		return bicircloid<Vec2>(f1,f2, 0.5f,0.5f);
	}

	// Get kth harmonic as new object
	PRamp harmonic(int k) const {
		return { mFreq*k, std::fmod(mPhase*k, 1.f) };
	}

private:
	float mPhase;	// current phase of ramp
	float mFreq;	// 1/period

	static constexpr float twoPi = 6.283185307179586;
};

#endif // include guard
