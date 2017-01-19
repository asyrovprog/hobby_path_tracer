#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <ctime>
#define die(why, with) if (why) { return (with); }
using namespace std;
using cfloat = const float;
struct vec3 { 
	float x, y, z; 
	vec3() {} vec3(cfloat& _x, cfloat& _y, cfloat& _z): x(_x), y(_y), z(_z) {}
	cfloat& operator[](int i) const { return (&x)[i]; }
	float& operator[](int i) { return (&x)[i];  }
};
const size_t width = 1024, height = (width * 2) / 3, path_count = 20000, raypix = 4, filt = 2; // size of output image, number of paths per image pixel
const vec3   eye = { 0, 0, 4 }; // eye location in world space
const float  F_EPSILON = 1e-7f, T_MAX = 1e2f, T_MIN = 1e-4f, F_PI = 3.1415926535897932384626433832795f, chessz = .33, ktex = .0; // eps/tmax/tmin are to mitigate float point math limits and div by 0
using cvec3  = const vec3;
using byte   = unsigned char;
vec3 operator+(cvec3& a, cvec3& b) { 
	return { a.x + b.x, a.y + b.y, a.z + b.z }; 
}
vec3 operator-(cvec3& a, cvec3& b) { 
	return{ a.x - b.x, a.y - b.y, a.z - b.z }; 
}
vec3 operator-(cvec3& v) {
	return{-v.x, -v.y, -v.z};
}
vec3 operator*(cfloat& f, cvec3& v) {
	return{ v.x * f, v.y * f, v.z * f };
}
vec3 operator*(cvec3& v, cfloat& f) {
	return f * v;
}
vec3 operator*(cvec3& v, cvec3& u) {
	return{ v.x * u.x, v.y * u.y, v.z * u.z };
}
vec3 operator/(cvec3& v, cfloat& f) {
	return{v.x / f, v.y / f, v.z / f};
}
float dot(cvec3& v, cvec3& u) {
	return v.x * u.x + v.y * u.y + v.z * u.z;
}
vec3 cross(cvec3& v, cvec3& u) {
	return{ v[1] * u[2] - u[1] * v[2], v[2] * u[0] - u[2] * v[0], v[0] * u[1] - u[0] * v[1] };
}
float length2(cvec3& v) {
	return dot(v, v);
}
float length(cvec3& v) {
	return sqrt(length2(v));
}
float map(cfloat& v, cfloat& x0, cfloat& x1, cfloat& y0, cfloat y1) {
	return y0 + (v - x0) * (y1 - y0) / (x1 - x0);
}
template <typename T> T clamp(const T& v, const T& a, const T& b) {
	return v < a ? a : (v > b ? b : v);
}
byte f2b(cfloat& v, cfloat& vmax = 1, cfloat& vmin = 0) {
	return (byte)round(map(clamp(v, vmin, vmax), vmin, vmax, 0.f, 255.f));
}
float rndf() { 
	static default_random_engine rndengl(clock()); 
	static uniform_real_distribution<float> rnddist(-1, 1); 
	return rnddist(rndengl); 
}
bool between(cfloat& v, cfloat& x0, cfloat& x1) { 
	return !(v < x0 || v > x1); 
}
cvec3 normalize(cvec3& v) { 
	return v * (1 / length(v)); 
}
vec3 ray2pix(size_t i, size_t j) { 
	return normalize(vec3(map(i + (rndf() + 1) / 2, 0, width, -1.5, 1.5), map(j + (rndf() + 1) / 2, 0, height, 1, -1), 0) - eye); 
}

bool save_images(const char* prf, const vector<vec3>& data, cfloat& vmin = 0.f, cfloat& vmax = 1.f) {
	auto f = &data[0].x;
	ofstream ppm(prf);
	die(!ppm.is_open(), false);
	ppm << "P3" << endl << width << ' ' << height << endl << 255 << endl;
	for (auto j = 0; j < data.size() * 3; j++)
		ppm << f2b(f[j], vmax, vmin) << (j % 3 == 2 ? '\n' : ' ');
	return true;
}

bool cone(cvec3& o, cvec3& d, vec3& hit) {
	auto a2 = 2 * (d.x * d.x + d.z * d.z - d.y * d.y);
	auto b = 2 * o.x * d.x + 2 * o.z * d.z - 2 * o.y * d.y;
	auto c = o.x * o.x + o.z * o.z - o.y * o.y;
	auto v = b * b - 2 * a2 * c;
	die(v < 0 || a2 < F_EPSILON, false);
	auto sv = sqrt(v);
	auto t1 = (-b + sv) / a2;
	auto t2 = (-b - sv) / a2;
	if (t1 < 0 || t2 > 0 && t2 < t1)
		t1 = t2;
	die(t1 < 0, false);
	hit = o + d * t1;
	return true;
}

int main(int argc, char** argv) { // single expected parameter is path + prefix of output images like "~/tmp/img-" would result if files like ~/tmp/img-555.ppm viewed with gimp
	vector<vec3> f(width * height); size_t j;
	vec3 hit1;
	#pragma omp parallel for private(j)
	for (j = 0; j < height; j++)
		for (auto i = 0; i < width; i++) {
			vec3 hit;
			auto dir = ray2pix(i, j);
			if (cone(eye, dir, hit)) {
				f[j * width + i] = { 1.f, 1.f, 1.f };
			} else {
				printf(".");
			}
				
		}
	save_images("c:\\local\\path_trace.ppm", f);
	return 1;
}

　
