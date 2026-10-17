/**
 * kiwi-orbslam-session: one monocular ORB-SLAM3 session over a directory of frames.
 *
 * Usage:
 *   kiwi-orbslam-session --mode map|localize
 *     --vocabulary PATH --settings PATH
 *     --sequence-dir DIR --timestamps PATH
 *     --pose-observations PATH
 *     [--atlas-out PATH.osa]     (map: the atlas is written on shutdown)
 *     [--atlas-in PATH.osa]      (localize: the atlas is not modified)
 *     [--mask-dir DIR]           (keypoint exclude masks, DIR/<t_ns>.png per frame)
 *
 * The sequence is DIR/<t_ns>.jpg for every line of the timestamps file (integer
 * nanoseconds). Pose observations are one row per input frame:
 * "t_ns state tx ty tz qx qy qz qw", camera-in-map pose, zeros when not tracked,
 * in both modes.
 *
 * This is a combined GPL-3.0 work (it links ORB-SLAM3); see LICENSE beside
 * this file and third_party/NOTICE.
 */

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sophus/se3.hpp>

#include <System.h>

using std::string;

int main(int argc, char **argv) {
    std::map<string, string> a;
    for (int i = 1; i + 1 < argc; i += 2) a[string(argv[i]).substr(2)] = argv[i + 1];

    const string mode = a["mode"];
    const string atlas = a[mode == "map" ? "atlas-out" : "atlas-in"];
    const size_t slash = atlas.rfind('/');
    const string atlas_dir = slash == 0 ? "/" : atlas.substr(0, slash);
    const string atlas_file = atlas.substr(slash + 1);

    std::vector<long long> t_ns;
    {
        std::ifstream f(a["timestamps"]);
        long long t;
        while (f >> t) t_ns.push_back(t);
    }

    // ORB-SLAM3 names its atlas with a settings key and reads or writes "./<name>.osa":
    // the engine runs in the atlas directory with a copy of the settings that carries
    // the key.
    if (chdir(atlas_dir.c_str()) != 0) return 1;
    const string settings = a["pose-observations"] + ".settings.yaml";
    {
        std::ifstream in(a["settings"]);
        std::ofstream out(settings);
        out << in.rdbuf() << "\n"
            << (mode == "map" ? "System.SaveAtlasToFile" : "System.LoadAtlasFromFile") << ": \""
            << atlas_file.substr(0, atlas_file.size() - 4) << "\"\n";
    }

    ORB_SLAM3::System SLAM(a["vocabulary"], settings, ORB_SLAM3::System::MONOCULAR,
                           /*bUseViewer=*/false);
    std::remove(settings.c_str());
    if (mode == "localize") SLAM.ActivateLocalizationMode();

    std::ofstream obs(a["pose-observations"]);
    obs << std::fixed << std::setprecision(9);

    for (size_t i = 0; i < t_ns.size(); i++) {
        const string name = std::to_string(t_ns[i]);
        cv::Mat im = cv::imread(a["sequence-dir"] + "/" + name + ".jpg", cv::IMREAD_UNCHANGED);
        if (!a["mask-dir"].empty())
            SLAM.SetNextKpMask(cv::imread(a["mask-dir"] + "/" + name + ".png",
                                          cv::IMREAD_GRAYSCALE));
        Sophus::SE3f Tcw = SLAM.TrackMonocular(im, t_ns[i] / 1e9);
        int state = SLAM.GetTrackingState();
        if (state == ORB_SLAM3::Tracking::OK) {
            Sophus::SE3f Twc = Tcw.inverse();
            Eigen::Vector3f p = Twc.translation();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            obs << t_ns[i] << " " << state << " " << p.x() << " " << p.y() << " "
                << p.z() << " " << q.x() << " " << q.y() << " " << q.z() << " "
                << q.w() << "\n";
        } else {
            obs << t_ns[i] << " " << state << " 0 0 0 0 0 0 0\n";
        }
    }
    obs.close();

    // Shutdown stops local mapping and loop closing and, in map mode, writes the atlas.
    SLAM.Shutdown();
    return 0;
}
