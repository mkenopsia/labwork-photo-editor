#include <iostream>
#include <vector>
#include <chrono>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include <string>
#include <filesystem>
#include <windows.h>
#include <fstream>
#include <algorithm>
#include <immintrin.h>

namespace fs = std::filesystem;

std::vector<std::string> filters = {
    "Solarize", "Posterize", "Sepia"
};

std::vector<std::string> mods = {
    "Single", "openMP", "SIMD"
};

void solarize(std::vector<unsigned char>& image, int width, int height, int channels) {
    for (int i = 0; i < width * channels * height; i++) {
        image[i] = (image[i] > 128) ? (255 - image[i]) : image[i];
    }
}

void solarize_openMP(std::vector<unsigned char>& image, int width, int height, int channels) {
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < width * channels * height; i++) {
            image[i] = (image[i] > 128) ? (255 - image[i]) : image[i];
        }
    }
}

void solarize_avx2(std::vector<unsigned char>& image, int width, int height, int channels) {
    const int size = width * height * channels;
    unsigned char* data = image.data();

    __m256i comparator = _mm256_set1_epi8(128);
    __m256i maxValue = _mm256_set1_epi8(255);

    for (int i = 0; i < size; i += 32) {
        __m256i values = _mm256_loadu_si256((__m256i*)(data + i));

        __m256i mask = _mm256_cmpgt_epi8(_mm256_subs_epu8(values, comparator), _mm256_setzero_si256());

        __m256i invertedValues = _mm256_sub_epi8(maxValue, values);
        __m256i result = _mm256_blendv_epi8(values, invertedValues, mask);

        _mm256_storeu_si256((__m256i*)(data + i), result);
    }

    for (int i = size - (size % 32); i < size; i++) {
        data[i] = (data[i] < 128) ? data[i] : 255 - data[i];
    }
}

void posterize(std::vector<unsigned char>& image, int width, int height, int channels) {
    int size = width * height * channels;
    for (int i = 0; i < size; i++) {
        image[i] = image[i] / 64 * 64;
    }   
}

void posterize_openMP(std::vector<unsigned char>& image, int width, int height, int channels) {
    int size = width * height * channels;
     #pragma omp parallel
    {
         #pragma omp for 
        for (int i = 0; i < size; i++) {
            image[i] = image[i] / 64 * 64;
        }
    }

}

void posterize_avx2(std::vector<unsigned char>& image, int width, int height, int channels) {
    const int size = width * height * channels;
    unsigned char* data = image.data();


    for (int i = 0; i < size; i += 32) {
        //image[i] = (image[i] / 64) * 64;

        __m256i values = _mm256_loadu_si256((__m256i*)(data + i));
        __m256i dividedValues = _mm256_srli_epi16(values, 6);
        __m256i result = _mm256_slli_epi16(dividedValues, 6);

        _mm256_storeu_si256((__m256i*)(data + i), result);
    }
    for (int i = size - (size % 32); i < size; i++) {
        data[i] = (data[i] / 64) * 64;
    }

}

void sepia(std::vector<unsigned char>& image, int width, int height, int channels) {
    for (int i = 0; i < width * height * channels; i += channels) {
        unsigned char red = image[i];
        unsigned char green = image[i + 1];
        unsigned char blue = image[i + 2];
        image[i] = (unsigned char)min(255, (int)(0.393 * red + 0.769 * green + 0.189 * blue));
        image[i + 1] = (unsigned char)min(255, (int)(0.349 * red + 0.686 * green + 0.168 * blue));
        image[i + 2] = (unsigned char)min(255, (int)(0.272 * red + 0.534 * green + 0.131 * blue));
    }
}

void sepia_openMP(std::vector<unsigned char>& image, int width, int height, int channels) {
#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < width * height * channels; i += channels) {
            unsigned char red = image[i];
            unsigned char green = image[i + 1];
            unsigned char blue = image[i + 2];
            image[i] = (unsigned char)min(255, (int)(0.393 * red + 0.769 * green + 0.189 * blue));
            image[i + 1] = (unsigned char)min(255, (int)(0.349 * red + 0.686 * green + 0.168 * blue));
            image[i + 2] = (unsigned char)min(255, (int)(0.272 * red + 0.534 * green + 0.131 * blue));
        }
    }
}
void sepia_avx2(std::vector<unsigned char>& image, int width, int height, int channels) {
    unsigned char* data = image.data();
    const int size = width * height * channels;
    __m256 coeff_r = _mm256_set1_ps(0.393f); // package single-precision
    __m256 coeff_g = _mm256_set1_ps(0.769f);
    __m256 coeff_b = _mm256_set1_ps(0.189f);
    __m256 coeff_r2 = _mm256_set1_ps(0.349f);
    __m256 coeff_g2 = _mm256_set1_ps(0.686f);
    __m256 coeff_b2 = _mm256_set1_ps(0.168f);
    __m256 coeff_r3 = _mm256_set1_ps(0.272f);
    __m256 coeff_g3 = _mm256_set1_ps(0.534f);
    __m256 coeff_b3 = _mm256_set1_ps(0.131f);
    __m256 max_val = _mm256_set1_ps(255.0f);

    int i = 0;
    for (; i < size; i += 24) {

        __m256i pixel_data = _mm256_loadu_si256((__m256i*)(data + i));

        // èçâëåêàåì öâåòà êðàñíûé, çåëåíûé, ñèíèé èç âåêòîðà ïèêñåëåé
        __m256i r = _mm256_and_si256(pixel_data, _mm256_set1_epi32(0xFF));
        __m256i g = _mm256_and_si256(_mm256_srli_epi32(pixel_data, 8), _mm256_set1_epi32(0xFF));
        __m256i b = _mm256_and_si256(_mm256_srli_epi32(pixel_data, 16), _mm256_set1_epi32(0xFF));

        // ïðåîáðàçóåì â float äëÿ âû÷èñëåíèé
        __m256 r_f = _mm256_cvtepi32_ps(r);
        __m256 g_f = _mm256_cvtepi32_ps(g);
        __m256 b_f = _mm256_cvtepi32_ps(b);

        // ôîðìóëà
        __m256 new_r = _mm256_add_ps(
            _mm256_mul_ps(r_f, coeff_r),
            _mm256_add_ps(
                _mm256_mul_ps(g_f, coeff_g),
                _mm256_mul_ps(b_f, coeff_b)
            )
        );

        __m256 new_g = _mm256_add_ps(
            _mm256_mul_ps(r_f, coeff_r2),
            _mm256_add_ps(
                _mm256_mul_ps(g_f, coeff_g2),
                _mm256_mul_ps(b_f, coeff_b2)
            )
        );

        __m256 new_b = _mm256_add_ps(
            _mm256_mul_ps(r_f, coeff_r3),
            _mm256_add_ps(
                _mm256_mul_ps(g_f, coeff_g3),
                _mm256_mul_ps(b_f, coeff_b3)
            )
        );

        // êàñò äî 255 âìåñòî ÿâíîãî ïðèâåäåíèÿ (unsigned char)
        new_r = _mm256_min_ps(new_r, max_val);
        new_g = _mm256_min_ps(new_g, max_val);
        new_b = _mm256_min_ps(new_b, max_val);

        // â int
        __m256i new_r_i = _mm256_cvtps_epi32(new_r);
        __m256i new_g_i = _mm256_cvtps_epi32(new_g);
        __m256i new_b_i = _mm256_cvtps_epi32(new_b);

        __m256i new_pixel = _mm256_or_si256(new_r_i, _mm256_slli_epi32(new_g_i, 8));
        new_pixel = _mm256_or_si256(new_pixel, _mm256_slli_epi32(new_b_i, 16));

        _mm256_storeu_si256((__m256i*)(data + i), new_pixel);
    }

    // Îáðàáîòêà îñòàâøèõñÿ ïèêñåëåé
    for (; i < size; i += channels) {
        unsigned char red = image[i];
        unsigned char green = image[i + 1];
        unsigned char blue = image[i + 2];
        image[i] = (unsigned char)min(255, (int)(0.393 * red + 0.769 * green + 0.189 * blue));
        image[i + 1] = (unsigned char)min(255, (int)(0.349 * red + 0.686 * green + 0.168 * blue));
        image[i + 2] = (unsigned char)min(255, (int)(0.272 * red + 0.534 * green + 0.131 * blue));
    }
}

std::string getPicPath(std::string path, std::string& str) {
    std::vector<std::string> fileNames;

    for (auto entry : fs::directory_iterator(path)) {
        if (entry.is_regular_file()) {
            fileNames.push_back(entry.path().filename().string());
        }
    }

    std::cout << "Âûáåðèòå êàðòèíêó äëÿ ðåäàêòèðîâàíèÿ:" << "\n";
    for (int i = 0; i < fileNames.size(); i++) {
        std::cout << i << ":  " << fileNames[i] << "\n";
    }
    int choose;
    std::cin >> choose;
    str = fileNames[choose];
    std::string findPath = R"(C:\Users\kaio\Desktop\orig\)" + fileNames[choose];
    return findPath;
}

int processing(std::vector<std::string> filters, std::vector<unsigned char>& image, int width, int height, int channels) {
    std::cout << "Âûáåðèòå ðåæèì: " << "\n";
    for (int i = 0; i < mods.size(); i++) {
        std::cout << i << "  " << mods[i] << "\n";
    }
    int chooseMode;
    std::cin >> chooseMode;

    std::cout << "Âûáåðèòå ôèëüòð: " << "\n";
    for (int i = 0; i < filters.size(); i++) {
        std::cout << i << "  " << filters[i] << "\n";
    }

    int chooseFilter;
    std::cin >> chooseFilter;

    switch (chooseMode) {
    case 0:
        switch (chooseFilter) {
        case 0:
            solarize(image, width, height, channels);
            break;
        case 1:
            posterize(image, width, height, channels);
            break;
        case 2:
            sepia(image, width, height, channels);
            break;
        default:;
        }
        break;
    case 1:
        switch (chooseFilter) {
        case 0:
            solarize_openMP(image, width, height, channels);
            break;
        case 1:
            posterize_openMP(image, width, height, channels);
            break;
        case 2:
            sepia_openMP(image, width, height, channels);
            break;
        default:;
        }
        break;

    case 2:
        switch (chooseFilter) {
        case 0:
            solarize_avx2(image, width, height, channels);
            break;
        case 1:
            posterize_avx2(image, width, height, channels);
            break;
        case 2:
            sepia_avx2(image, width, height, channels);
            break;
        default:;
        }
        break;
    }
    
    return chooseFilter;
}

int testCount = 100;

int main() {
    setlocale(LC_ALL, "RUS");
    SetConsoleOutputCP(CP_UTF8);

    std::string path = R"()"; // folder with photo to edit

    std::string picName;
    std::string findPath = getPicPath(path, picName);

    int width, height, channels;
    unsigned char* img = stbi_load(findPath.c_str(), &width, &height, &channels, 0);
    std::vector image(img, img + width * height * channels);

    int filterNum = processing(filters, image, width, height, channels);

    std::string resPath = R"()" + filters[filterNum] + picName; // path to save edited photo
    if (!stbi_write_jpg(resPath.c_str(), width, height, channels, image.data(), 90)) {
        std::cout << "Îøèáêà ñîõðàíåíèÿ èçîáðàæåíèÿ" << std::endl;
        stbi_image_free(img);
        return 1;
    }

    stbi_image_free(img);
    return 0;

}

// test system

//int main() {
//    SetConsoleOutputCP(CP_UTF8);
//
//    std::string path = R"(C:\Users\kaio\Desktop\orig\)";
//    std::vector<std::string> fileNames;
//    for (auto entry : fs::directory_iterator(path)) {
//        if (entry.is_regular_file()) {
//            fileNames.push_back(entry.path().filename().string());
//        }
//    }
//    for (int file = 0; file < fileNames.size(); file++) {
//        std::string findPath = R"(C:\Users\kaio\Desktop\orig\)" + fileNames[file];
//
//        std::string picName;
//
//        int width, height, channels;
//        unsigned char* img = stbi_load(findPath.c_str(), &width, &height, &channels, 0);
//        if (!img) {
//            std::cerr << "Îøèáêà çàãðóçêè èçîáðàæåíèÿ: " << findPath << "\n";
//            return 1;
//
//        }
//        std::vector<unsigned char> image(img, img + width * height * channels);
//        double time = 0;
//        for (int i = 0; i < testCount; i++) {
//            std::vector<unsigned char> copy(image);
//            auto start = std::chrono::high_resolution_clock::now();
//
//            sepia_avx2(image, width, height, channels);
//
//            auto end = std::chrono::high_resolution_clock::now();
//            std::chrono::duration<double> duration = end - start;
//            time += duration.count();
//            copy.clear();
//        }
//        std::cout << fileNames[file] << ": " << time / testCount << "\n";
//        stbi_image_free(img);
//    }
//    return 0;
//
//}
