#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#include <iostream>
#include <map>
#include <vector>
#include <stack>
#include <cassert>

struct lessVec3b {
    bool operator()(cv::Vec3b const& lhs, cv::Vec3b const& rhs) const {
        return (lhs[2] != rhs[2]) ? (lhs[2] < rhs[2])
                                  : ((lhs[1] != rhs[1]) ? (lhs[1] < rhs[1])
                                                        : (lhs[0] < rhs[0]));
    }
};

typedef std::map<cv::Vec3b, uint64_t, lessVec3b> map_type;

// https://stackoverflow.com/a/34734939/5008845
void reduce_colors_kmeans(cv::Mat3b const& src, cv::Mat3b& dst, int num_colors,
                          std::vector<int>& labels, cv::Mat1f& colors,
                          map_type& palette) {
    int n = src.rows * src.cols;
    cv::Mat data = src.reshape(1, n);
    data.convertTo(data, CV_32F);

    labels.reserve(n);
    cv::kmeans(data, num_colors, labels, cv::TermCriteria(), 1,
               cv::KMEANS_PP_CENTERS, colors);

    for (int i = 0; i < n; ++i) {
        int label = labels[i];
        data.at<float>(i, 0) = colors(label, 0);
        data.at<float>(i, 1) = colors(label, 1);
        data.at<float>(i, 2) = colors(label, 2);
        cv::Vec3b color(colors(label, 0), colors(label, 1), colors(label, 2));
        if (palette.count(color) == 0) {
            palette[color] = 1;
        } else {
            palette[color] += 1;
        }
    }

    cv::Mat reduced = data.reshape(3, src.rows);
    reduced.convertTo(dst, CV_8U);
}

typedef std::pair<float, float> point_type;

void print_color_numbers(cv::Mat3b& dst, int rows, int cols,
                         std::vector<int> const& labels,
                         std::vector<point_type> const& centroids,
                         cv::Mat1f const& colors, map_type& palette) {
    static const float font_scale = 1.0;
    static const float thickness = 0.5;

    assert(labels.size() == centroids.size());

    static const size_t step = 1;
    for (size_t i = 0; i < centroids.size(); i += step) {
        if (i > centroids.size()) break;
        auto centroid = centroids[i];
        int label = labels[i];

        cv::Vec3b color(colors(label, 0), colors(label, 1), colors(label, 2));

        auto it = palette.find(color);
        assert(it != palette.end());

        auto color_id = (*it).second;
        std::cout << "color_id " << color_id << std::endl;

        std::cout << "putting centroid " << i << std::endl;
        cv::putText(dst, std::to_string(color_id),
                    cv::Point(centroid.second, centroid.first),
                    cv::FONT_HERSHEY_PLAIN, font_scale, CV_RGB(255, 0, 0),
                    thickness);
    }
}

void reshape2d(std::vector<int> const& labels,
               std::vector<std::vector<int>>& labels2d, int num_rows,
               int num_cols) {
    assert(labels.size() == num_rows * num_cols);
    labels2d.resize(num_rows, std::vector<int>());
    for (int i = 0; i != num_rows; ++i) {
        labels2d[i].resize(num_cols);
        memcpy(&labels2d[i][0], &labels[i * num_cols], sizeof(int) * num_cols);
    }
}

void visit(std::vector<std::vector<int>> const& labels2d, int r, int c,
           int label, std::vector<std::vector<bool>>& visited,
           std::vector<point_type>& region) {
    assert(r >= 0);
    assert(c >= 0);
    assert(r < labels2d.size());
    assert(c < labels2d[0].size());
    // std::cout << "visiting [" << r << "," << c << "]" << std::endl;
    assert(labels2d[r][c] == label);

    if (visited[r][c]) return;
    // visited[r][c] = true;

    // visit neighbours if they are unvisited and have the same label

    std::stack<point_type> queue;
    queue.emplace(r, c);

    while (!queue.empty()) {
        auto point = queue.top();
        r = point.first;
        c = point.second;
        queue.pop();
        if (!visited[r][c]) {
            visited[r][c] = true;
            region.emplace_back(r, c);

            // r-1,c-1
            if (r - 1 >= 0 and c - 1 >= 0 and !visited[r - 1][c - 1] and
                labels2d[r - 1][c - 1] == label) {
                queue.emplace(r - 1, c - 1);
            }

            // r-1,c
            if (r - 1 >= 0 and !visited[r - 1][c] and
                labels2d[r - 1][c] == label) {
                queue.emplace(r - 1, c);
            }

            // r-1,c+1
            if (r - 1 >= 0 and c + 1 < labels2d[0].size() and
                !visited[r - 1][c + 1] and labels2d[r - 1][c + 1] == label) {
                queue.emplace(r - 1, c + 1);
            }

            // r,c-1
            if (c - 1 >= 0 and !visited[r][c - 1] and
                labels2d[r][c - 1] == label) {
                queue.emplace(r, c - 1);
            }

            // r,c+1
            if (c + 1 < labels2d[0].size() and !visited[r][c + 1] and
                labels2d[r][c + 1] == label) {
                queue.emplace(r, c + 1);
            }

            // r+1,c-1
            if (r + 1 < labels2d.size() and c - 1 >= 0 and
                !visited[r + 1][c - 1] and labels2d[r + 1][c - 1] == label) {
                queue.emplace(r + 1, c - 1);
            }

            // r+1,c
            if (r + 1 < labels2d.size() and !visited[r + 1][c] and
                labels2d[r + 1][c] == label) {
                queue.emplace(r + 1, c);
            }

            // r+1,c+1
            if (r + 1 < labels2d.size() and !visited[r + 1][c + 1] and
                c + 1 < labels2d[0].size() and
                labels2d[r + 1][c + 1] == label) {
                queue.emplace(r + 1, c + 1);
            }
        }
    }
}

void visit(std::vector<std::vector<int>> const& labels2d,
           std::vector<std::vector<point_type>>& regions) {
    int num_rows = labels2d.size();
    int num_cols = labels2d[0].size();

    std::vector<std::vector<bool>> visited;
    visited.resize(num_rows, std::vector<bool>());
    for (int r = 0; r != num_rows; ++r) { visited[r].resize(num_cols, false); }

    for (int r = 0; r != num_rows; ++r) {
        for (int c = 0; c != num_cols; ++c) {
            std::vector<point_type> region;
            int label = labels2d[r][c];
            visit(labels2d, r, c, label, visited, region);
            if (region.size() > 0) { regions.push_back(region); }
        }
    }
}

point_type compute_centroid(std::vector<point_type> const& polygon) {
    double sum_x = 0.0;
    double sum_y = 0.0;
    for (auto x : polygon) {
        sum_x += x.first;
        sum_y += x.second;
    }
    return {sum_x / polygon.size(), sum_y / polygon.size()};
}

void auto_canny(cv::Mat3b& src, cv::Mat& canny, std::string const& filename) {
    cv::Mat detected_edges;
    Canny(src, detected_edges, 1, 1, 3);
    canny = ~detected_edges;

    // uncomment the following piece of code to blend edges and original image
    // cv::Mat dst = Scalar::all(0);
    // cv::Mat addweight;
    // src.copyTo(dst, detected_edges);  // copy part of src image according to
    // the
    //                                   // canny output, canny is used as mask
    // cvtColor(detected_edges, detected_edges,
    //          COLOR_GRAY2BGR);  // convert canny image to bgr
    // addWeighted(src, 1.0, detected_edges, 1.0, 1.0,
    //             addweight);  // blend src image with canny image
    // src += detected_edges;   // add src image with canny image
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << argv[0] << " [path_to_file] [num_colors] [blur_level]"
                  << std::endl;
        return 1;
    }

    std::string filename(argv[1]);
    int num_colors = std::atoi(argv[2]);
    int blur_level = std::atoi(
        argv[3]);  // typical values are 7, 9, or larger for bigger images
    cv::Mat3b read = cv::imread(argv[1]);

    // by enlarging the image, we need more blur
    cv::Mat3b img;
    cv::resize(read, img, cv::Size(), 2.0, 2.0);

    // smooth image to remove noise
    cv::Mat3b blurred;
    cv::medianBlur(img, blurred, blur_level);

    // apply bilateral filtering
    cv::Mat3b filtered;
    cv::bilateralFilter(blurred, filtered, 9, 17, 17);

    // reduce number of colors
    cv::Mat3b reduced;
    std::vector<int> labels;
    cv::Mat1f colors;
    map_type palette;
    reduce_colors_kmeans(filtered, reduced, num_colors, labels, colors,
                         palette);

    static const double MIN_AREA = 0.1;
    std::cout << "rows: " << img.rows << std::endl;
    std::cout << "cols: " << img.cols << std::endl;
    uint64_t area = img.rows * img.cols;

    uint64_t count = 0;
    for (auto& color : palette) {
        double color_area = (color.second * 100.0) / area;
        if (color_area > MIN_AREA) ++count;
    }

    static const uint64_t PALETTE_SIZE = 150;
    static const uint64_t OFFSET = 50;
    // Print palette and assing color id
    cv::Mat3b palette_img(count * PALETTE_SIZE, PALETTE_SIZE + OFFSET);
    palette_img.setTo(cv::Scalar(255, 255, 255));
    uint64_t id = 1;
    for (auto& color : palette) {
        double color_area = (color.second * 100.0) / area;
        if (color_area > MIN_AREA) {
            color.second = id;

            std::cout << "Color " << color.second << " : rgb("
                      << int(color.first[2]) << ", " << int(color.first[1])
                      << ", " << int(color.first[0]) << ")"
                      << " \t - Area: " << color_area << "%" << std::endl;

            uint64_t base = (id - 1) * PALETTE_SIZE;
            for (uint64_t i = 0; i != PALETTE_SIZE; ++i) {
                for (uint64_t j = 0; j != PALETTE_SIZE; ++j) {
                    palette_img.at<cv::Vec3b>(base + i, j + OFFSET) =
                        color.first;
                }
                cv::putText(palette_img, std::to_string(id),
                            cv::Point(0, base + PALETTE_SIZE / 2),
                            cv::FONT_HERSHEY_PLAIN, 2, CV_RGB(0, 0, 0), 2);
            }

            id += 1;
        }
    }

    cv::imwrite(filename + ".palette.jpeg", palette_img);

    std::vector<std::vector<int>> labels2d;
    reshape2d(labels, labels2d, img.rows, img.cols);

    std::vector<std::vector<point_type>> regions;
    visit(labels2d, regions);

    std::cout << "visit done!" << std::endl;

    // Canny's edge detection
    cv::Mat canny;
    auto_canny(reduced, canny, filename);
    cv::imwrite(filename + ".canny.jpeg", canny);
    cv::imwrite(filename + ".toonified.jpeg", reduced);
    cv::Mat3b canny2 = cv::imread((filename + ".canny.jpeg").c_str());

    std::vector<point_type> centroids;
    centroids.reserve(regions.size());
    std::vector<int> final_labels;
    final_labels.reserve(regions.size());

    uint64_t sum = 0;
    for (uint64_t i = 0; i != regions.size(); ++i) {
        auto const& region = regions[i];
        sum += region.size();
        double color_area = (region.size() * 100.0) / area;
        // std::cout << color_area << std::endl;
        if (color_area > MIN_AREA / 10) {
            // std::cout << region.size() << " points in region" << std::endl;
            auto centroid = compute_centroid(region);
            centroids.push_back(centroid);
            auto first_point = region.front();
            uint64_t index = first_point.first * img.cols + first_point.second;
            int label = labels[index];

            // for (auto p : region) {
            //     assert(labels[p.first * img.cols + p.second] == label);
            // }

            final_labels.push_back(label);

            // cv::Vec3b color(colors(label, 0), colors(label, 1),
            //                 colors(label, 2));
            // std::cout << "  color is: "
            //           << "rgb(" << int(color[2]) << ", " << int(color[1])
            //           << ", " << int(color[0]) << ")" << std::endl;

            // static const float font_scale = 1.0;
            // static const float thickness = 0.5;
            // cv::Mat3b tmp = canny2.clone();
            // for (auto point : region) {
            //     tmp.at<cv::Vec3b>(point.first, point.second) =
            //         cv::Vec3b(255, 0, 0);
            // }
            // imwrite("./annotated" + std::to_string(i) + ".jpeg", tmp);
        }
    }

    assert(sum == area);
    std::cout << "#ignore " << sum << std::endl;

    cv::Mat3b annotated = canny2.clone();
    print_color_numbers(annotated, img.rows, img.cols, final_labels, centroids,
                        colors, palette);
    imwrite(filename + ".annotated.jpeg", annotated);

    return 0;
}