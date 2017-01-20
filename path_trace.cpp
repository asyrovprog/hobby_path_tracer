#include <fstream>
#include <cfloat>
#include <string>
#include <random>
#include <algorithm>
using namespace std; using cfloat = const float;  using byte = unsigned char;
struct vec3 {
    float x, y, z;
    float& operator[](int i)        { return (&x)[i]; }
    cfloat& operator[](int i) const { return (&x)[i]; }
}; using cvec3 = const vec3;
struct material_t { // type: 0 diffusive, 1 mirror, 2 transparent, 3 phong; pn: cosine power for phong, emission: for diffuse or phong; tex: chessboard texture for diffuse or phong - 0 none, 1 xz (floor), 2 yz (side), 3 xy (back)
    int type; vec3 color; float pn, emission; int tex;
}; using cmaterial_t = const material_t;
struct shape_t { // type: 0 sphere, 1 plane; cn: center for sphere or normal for plane; r: radius for sphere, distance to origin for plane
    byte type; vec3 cn, r; material_t m;
}; using cshape_t = const shape_t;
const size_t width = 1024, height = (width * 2) / 3, path_count = 5000, raypix = 2, filt = 2; // size of output image, number of paths per image pixel
const vec3   eye = {0, 0, 4}; // eye location in world space
const float  F_EPSILON = 1e-7f, T_MAX = 1e2f, T_MIN = 1e-4f, F_PI = 3.1415926535897932384626433832795f, chessz = .33, ktex = .0; // eps/tmax/tmin are to mitigate float point math limits and div by 0
#define die(why, with)               if (why) { return (with); }
float pow2(cfloat& x)                { return x * x; }
cvec3 operator*(cvec3& v, cfloat& f) { return {v.x * f, v.y * f, v.z * f}; }
cvec3 operator*(cfloat& f, cvec3& v) { return v * f; }
cvec3 operator/(cvec3& v, cfloat& f) { return {v.x / f, v.y / f, v.z / f}; }
cvec3 operator+(cvec3& v, cvec3& u)  { return {v.x + u.x, v.y + u.y, v.z + u.z}; }
cvec3 operator-(cvec3& v, cvec3& u)  { return {v.x - u.x, v.y - u.y, v.z - u.z}; }
cvec3 operator-(cvec3& v)            { return {-v.x, -v.y, -v.z}; }
cvec3 operator*(cvec3& v, cvec3& u)  { return {v.x * u.x, v.y * u.y, v.z * u.z}; }
float vmax(cvec3 &v, cfloat& a)      { return max(a, max(v.x, max(v.y, v.z))); }
float dot(cvec3& v, cvec3& u)        { return v.x * u.x + v.y * u.y + v.z * u.z; }
float length(cvec3& v)               { return sqrt(dot(v, v)); }
cvec3 normalize(cvec3& v)            { return v * (1 / length(v)); }
cvec3 reflect(cvec3& i, cvec3& n)    { return i - 2 * n * dot(n, i); }
float brightness(cvec3& p)           { return sqrt(0.299f * pow2(p.x) + 0.587f * pow2(p.y) + 0.114f * pow2(p.z)); } // http://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color
float clamp(cfloat& v, cfloat& a = 0, cfloat& b = 1) { return v > b ? b : (v < a ? a : v); }
bool refract(cvec3& i, cvec3& n, cfloat& ir, vec3& r) {
    double cos_i = dot(-i, n), cos_t2 = 1.f - pow2(ir) * (1.f - pow2(cos_i));
    die(cos_t2 < 0.f, false);
    r = ir * i + ((ir * cos_i - sqrt(abs(cos_t2))) * n);
    return true;
}
float fresnel_schlick(cvec3& i, cvec3& n, cfloat& ni, cfloat& nt) { // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations
    auto cos_x = clamp(dot(-i, n));
    if (ni > nt) {
        auto sin_t2 = pow2(ni / nt) * (1 - pow2(cos_x));
        die(sin_t2 > 1, 1);
        cos_x = sqrt(1 - sin_t2);
    }
    auto r0 = pow2((ni - nt) / (nt + ni));
    return r0 + (1 - r0) * pow(1 - cos_x, 5);
}
bool ray_plane(cfloat& d, cvec3& pd, cvec3& rp, cvec3& rd, cfloat& tmin, cfloat& tmax, vec3& h, vec3& n, float& it) {
    float t = dot(rd, pd);
    die(abs(t) < F_EPSILON, false);
    t = (d - dot(rp, pd)) * (1 / t);
    die(t < tmin || t > tmax, false);
    h  = rp + t * rd; n = pd; it = t;
    return true;
}
bool ray_sphere(cvec3& sc, cfloat& sr, cvec3& p, cvec3& pd, cfloat& tmin, cfloat& tmax, vec3& h, vec3& n, float& it) {
    float a2 = dot(pd, pd) * 2., b = 2. * dot(pd, p - sc), c = dot(sc, sc) + dot(p, p) - 2. * dot(sc, p) - sr * sr, d = b * b - 2. * a2 * c, t;
    die(d < 0, false);
    d = sqrt(d);
    if ((t = (-b - d) / a2) < tmin)
        t = (-b + d) / a2;
    die(t < tmin || t > tmax, false);
    h = p + t * pd; n = (h - sc) / sr; it = t;
    return true;
}
float rndf() { static default_random_engine rndengl(clock()); static uniform_real_distribution<float> rnddist(-1, 1); return rnddist(rndengl); }
bool  between(cfloat& v, cfloat& x0, cfloat& x1) { return !(v < x0 || v > x1); }
float map(cfloat& v, cfloat& x1, cfloat& x2, cfloat& y1, cfloat& y2) { return y1 + (v  - x1) * (y2 - y1) * (1 / (x2 - x1)); }
int   f2b(float v, float cmax) { return (int) round(map(clamp(v, 0, cmax), 0, cmax, 0, 255)); }
vec3  ray2pix(size_t i, size_t j) { return normalize(vec3({map(i + (rndf() + 1) / 2, 0, width, -1.5, 1.5), map(j + (rndf() + 1) / 2, 0, height, 1, -1), 0}) - eye); }
float texture(cmaterial_t& m, cvec3& h) { if(m.tex == 0) { return 1; } float a = (m.tex == 2 ? h.y : h.x) + 9.7f, b = (m.tex == 3 ? h.y : h.z) + 9.3f; return ((int)(a / chessz) % 2 == 0 ^ (int)(b / chessz) % 2 == 1) ? 1 : ktex; }
vec3  brdf(cmaterial_t& m, cvec3& i, cvec3& o, cvec3& n) { return m.type == 0 ? m.color / F_PI : m.color * (pow(clamp(dot(reflect(i, n), o)), m.pn) * (2 + m.pn)/(2 * F_PI)); }
bool  intersect(cshape_t& s, cvec3& ro, cvec3& rd, cfloat t0, cfloat t1, vec3& h, vec3& n, float& t) { return (s.type == 0) ? ray_sphere(s.cn, s.r.x, ro, rd, t0, t1, h, n, t) : ray_plane(s.r.x, s.cn, ro, rd, t0, t1, h, n, t); }
vector<shape_t> shapes = {
    // room: bottom, up, left, right, front, back planes
    { 1, {0, 1, 0}, {-1, 0, 0}, { 1, {1, 1, 1}, 1, 0, 1 } }, { 1, {0,-1, 0}, {-1, 0, 0}, { 0, {1, 1, 1}, 1, 0, 0 } }, { 1, {1, 0, 0}, {-1.5, 0, 0}, { 0, {1, 0.7, 0.7}, 1, 0, 0 } },
    { 1, {-1, 0, 0}, {-1.5, 0, 0}, { 0, {0.7, 0.7, 1}, 1, 0, 0 } }, { 1, {0, 0, 1}, {-3, 0, 0}, { 0, {1, 1, 1}, 1, 0, 0 } }, { 1, {0, 0,-1}, {-6.5, 0, 0}, { 0, {0, 0, 0}, 1, 0, 0 } },
    { 0, {-.5 ,-.65, -1.5}, {.35, 0, 0}, { 0, {.5, 1, .5}, 1, 0, 0 } }, // lambertian, green sphere
    { 0, {-1.1,-.8,-.4}, {.19, 0, 0}, { 0, {1, .5, .5}, 1, .1, 0 } },   // secondary minor pink light sphere on the floor
    { 0, {-1,-.25,-1}, {.3, 0, 0}, { 1, {0, 0, 0}, 1, 0, 0 } },         // reflective mirror sphere
    { 0, {0.9,-.2,-.8}, {.4, 0, 0}, { 2, {0, 0, 0}, 0, 0, 0 } },        // transparent glass/diamont sphere
    { 0, {0, 1.4, -1}, {.55, 0, 0}, { 0, {1, 1, 1}, 1, 1, 0 } },        // main ceiling light sphere
    { 0, {.4,-.55,-2}, {.45, 0, 0}, { 3, {1, .5, 0}, 6, 0, 0 } }        // phong with cosine power 6 sphere
}; /*couple of transparant spheres to observe dispersion (need to uncomment glass/diamont 'iors') and caustics */ // vector<shape_t> shapes = { { 1, {0, 1, 0}, {-1, 0, 0}, { 0, {1, 1, 1}, 1, 0, 0} }, { 1, {0, 0, 1}, {-2, 0, 0}, { 0, {1, 1, 1}, 1, 0, 0 } }, { 0, {-.2, -.45, -1}, {.5, 0, 0}, { 2, {1, 1, 1}, 0, 0, 0 } },{ 0, {.4, .0, 0}, {.6, 0, 0}, { 2, {1, 1, 1}, 0, 0, 0 } }, { 0, {2.2, 1.1, 2}, {0.75, 0, 0}, { 0, {1, 1, 1}, 1, 1, 0 } }};
vector<vec3>  clrs = {{1,1,1}}; // clrs = {{.26, 0, 0}, {.26, .13, 0}, {.26, .26, 0}, {0, .26, 0}, {0, .35, .35}, {0, 0, .26}, {.08, 0, .13}, {.14, 0, .26}};
vector<float> iors = {1.515};  // diamont: */ /*iors = {2.40735, 2.415, 2.41734, 2.42694, 2.43, 2.44, 2.452, 2.46476};*/ /* glass: */ iors = {1.509, 1.51, 1.511, 1.515, 1.516, 1.517, 1.519, 1.521};
void sample_hemisphere(cvec3& n, vec3& dir, float& cos_t) {
    auto phi = (rndf() + 1.f) * F_PI, cost = rndf(), sint = sin(acos(cost));
    dir = { sint * cos(phi), sint * sin(phi), cost };
    if ((cos_t = dot(dir, n)) < 0) {
        dir = -dir;
        cos_t = -cos_t;
    }
}
void sample_transmit(cvec3& rd, vec3& hn, cfloat& ior, vec3& pd, float& k, bool& in) {
    auto cost = dot(rd, hn); k = 0.f;
    if (!(cost > 0 && !in)) { // this check blocks float precision issue in intersection calculations
        in = cost > 0;
        auto ior1 = in ? ior : 1.f, ior2 = in ? 1.f : ior; hn = in ? -hn : hn;
        auto tir = !refract(rd, hn, ior1 / ior2, pd);
        auto p_refl = tir ? 1 : (k = fresnel_schlick(rd, hn, ior1, ior2));
        if ((rndf() + 1.f) / 2.f <= p_refl) {
            pd = reflect(rd, hn);
            k = tir ? 1 : k * (1.f / p_refl);
        } else {
            k = (1.f - k) * (1.f / (1.f - p_refl));
            in = !in;
        }
    }
}
vec3 trace(cvec3& ro, cvec3& rd, int clr, int inid = -1, float weight = 1, int depth = 0) {
    vec3 h, hn, pd; float ht = T_MAX, k = 1, t0 = T_MIN, p_emit, fc = 1.f;
    cmaterial_t *m = nullptr; int idx = -1; bool in = (inid != -1);
    for (int i = 0; i < shapes.size(); ++i)
        if ((inid == -1 || inid == i) && intersect(shapes[i], ro, rd, t0, ht, h, hn, ht))
            m = &shapes[(idx = i)].m;
    die(m == nullptr || ++depth > 256, vec3({0, 0, 0}));
    if (m->type == 0 || m->type == 3) {
        if ((rndf() + 1) / 2 <= (p_emit = m->emission * .6 + (1 - weight) * .3 + .1)) {
            return m->color * m->emission * (1 / p_emit);
        } else {
            sample_hemisphere(hn, pd, k);
            return trace(h, pd, clr, -1, k, depth) * brdf(*m, rd, pd, hn) * F_PI * k * (1 / (1 - p_emit)) * texture(*m, h);
        }
    } else if (m->type == 1) { // mirror, no use of brdf here, but consider mirror as ray redirection
        return trace(h, reflect(rd, hn), clr, -1, 1, depth) * texture(*m, h);
    }
    sample_transmit(rd, hn, iors[clr], pd, k, in); // transparent glass, again no brdf, russian rulette to reflect or refract
    return (k == 0 ? vec3({0, 0, 0}) : trace(h, pd, clr, (in ? idx : -1), 1, depth) * k);
}
float save_images(const char* prf, vector<vec3>& f) { // simple histogram based thresholding https://en.wikipedia.org/wiki/Image_histogram#Image_manipulation_and_histograms
    vector<size_t> h(50000, 0);
    float          step = 1.f / h.size(), inc = 150, cut = 0.935, *pf = &f[0].x, hid = 0;
    for (int j = 0; j <= f.size() * 3; j++)
        h[(size_t) clamp(pf[j] / step, 0, h.size() - 1)]++;
    for (size_t j = 0, t = 0; j < h.size() && t < (size_t) (width * height * 3. * cut); j++, hid++)
        t += h[j];
    for (size_t j = 0; j < 9 && hid < h.size(); j++, hid += inc * j) {
        ofstream ppm((string(prf) + "-" + to_string(width) + "-" + to_string(path_count) + "-" + to_string(j) + ".ppm"));
        ppm << "P3" << endl << width << ' ' << height << endl << 255 << endl;
        for (int j = 0; j < f.size() * 3; j++)
            ppm << f2b(pf[j], hid * step) << (j % 3 == 2 ? '\n' : ' ');
    }
}
void filter_frame(vector<vec3>& f, size_t wsz) { // median filter to reduce noise, note: also creates some dark black border of esz/2 size as side effect
    vector<vec3> ff(f.size(), vec3({0, 0, 0}));
    for (size_t esz = wsz / 2, j = esz, mid_id = (wsz * wsz) / 2; j < height - esz; j++)
        for (size_t i = esz; i < width - esz; i++) {
            vec3 wnd[wsz * wsz];
            for (size_t r = 0, t = 0; r < wsz; r++)
                for (size_t c = 0; c < wsz; c++)
                    wnd[t++] = f[(j + r - esz) * width + i + c - esz];
            sort(wnd, wnd + wsz * wsz, [&](cvec3& a, cvec3& b) { return brightness(a) < brightness(b); });
            auto value = f[j * width + i], med = wnd[mid_id];
            auto vlum  = brightness(value);
            ff[j * width + i] = vlum <= 0 ? med : value * (brightness(med) / vlum);
        }
    f = std::move(ff);
}
int main(int argc, char** argv) { // single expected parameter is path + prefix of output images like "~/tmp/img-" would result if files like ~/tmp/img-555.ppm viewed with gimp
    vector<vec3> f(width * height); size_t j;
    #pragma omp parallel for private(j)
    for (j = 0; j < height; j++)
        for (auto i = 0; i < width; i++) {
            auto c = vec3({0, 0, 0});
            for (auto s = 0; s < raypix; s++) {
                auto dir = ray2pix(i, j);
                for (auto k = 0; k < path_count / raypix; k++)
                    for (auto l = 0; l < clrs.size(); l++)
                        c = c + trace(eye, dir, l) * clrs[l];
            }
            f[j * width + i] = isnan(c.x + c.y + c.z) ? vec3({1, 0, 0}) : c / path_count;
        }
    if (filt > 1)
        filter_frame(f, min(height, filt));
    save_images((argc < 2 ? "~/path_trace-" : argv[1]), f);
    return 1;
}
