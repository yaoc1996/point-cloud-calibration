#define NOMINMAX

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Dense>

#include "lasreader.hpp"
#include "laswriter.hpp"
#include "kdtree.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define LAS_SAMPLING_RATE 100
#define ANGLE_THRESH M_PI * 20 / 180
#define NBR_QUERY_SIZE 50
#define DIST_THRESH 3

struct Options
{
    std::filesystem::path inputDir;
    std::filesystem::path outputDir;
    std::filesystem::path controlFile;
    int samplingRate = LAS_SAMPLING_RATE;
    float angleThresh = ANGLE_THRESH;
    float distThresh = DIST_THRESH;
    int maxThreads = std::thread::hardware_concurrency();
};

struct LASFileStats
{
    std::filesystem::path file;
    std::vector<int> controlPointIndices;
    KDTree<float> kdtree;
    float dz;
};

class Logger
{
public:
    Logger(const char *out, bool silent);

    void close();
    template <typename T>
    Logger &operator<<(const T &data);

private:
    std::ofstream outStream;
    bool silent;
};

Logger::Logger(const char *out, bool silent)
{
    this->outStream = std::ofstream(out);
    this->silent = silent;
}

void Logger::close()
{
    this->outStream.close();
}

template <typename T>
Logger &Logger::operator<<(const T &data)
{
    if (!silent)
    {
        std::cout << data << std::flush;
    }

    this->outStream << data << std::flush;
    return *this;
}

void usage(int argc, const char *argv[])
{
    std::cout << "Usage:" << std::endl;
    std::cout << "\t" << argv[0] << " -i <input_directory> -c <control_file> [...options]" << std::endl;
    std::cout << "\t" << argv[0] << " -h for more help" << std::endl;
}

void help()
{
    std::cout << "Required:" << std::endl;
    std::cout << "\t" << std::left << std::setw(10) << "-i"
              << "input directory" << std::endl;
    std::cout << "\t" << std::left << std::setw(10) << "-c"
              << "control file" << std::endl;
    std::cout << "Optional:" << std::endl;
    std::cout << "\t" << std::left << std::setw(10) << "-o"
              << "output directory (default: <input_directory>/calibrated)" << std::endl;
    std::cout << "\t" << std::left << std::setw(10) << "-r"
              << "sampling rate for computing calibration value (default: " << LAS_SAMPLING_RATE << ")" << std::endl;
    std::cout << "\t" << std::left << std::setw(10) << "-a"
              << "angle threshold in degrees for control point detection (default: " << (int)round(ANGLE_THRESH * 180 / M_PI) << ")" << std::endl;
    std::cout << "\t" << std::left << std::setw(10) << "-d"
              << "distance threshold in feet for control point detection (default: " << DIST_THRESH << ")" << std::endl;
    std::cout << "\t" << std::left << std::setw(10) << "-t"
              << "maximum number of concurrent processing threads (default: " << std::thread::hardware_concurrency() << ")" << std::endl;
}

void parseArgs(int argc, const char *argv[], Options *options)
{
    int i = 1;

    while (i < argc)
    {
        const char *arg = argv[i];

        if (strcmp(arg, "-i") == 0)
        {
            options->inputDir = std::filesystem::canonical(std::filesystem::path(argv[i + 1]));
        }
        else if (strcmp(arg, "-c") == 0)
        {
            options->controlFile = std::filesystem::canonical(std::filesystem::path(argv[i + 1]));
        }
        else if (strcmp(arg, "-o") == 0)
        {
            options->outputDir = std::filesystem::path(argv[i + 1]);
        }
        else if (strcmp(arg, "-r") == 0)
        {
            options->samplingRate = std::stoi(argv[i + 1]);
        }
        else if (strcmp(arg, "-a") == 0)
        {
            options->angleThresh = std::stof(argv[i + 1]) / 180 * M_PI;
        }
        else if (strcmp(arg, "-d") == 0)
        {
            options->distThresh = std::stof(argv[i + 1]);
        }
        else if (strcmp(arg, "-t") == 0)
        {
            options->maxThreads = std::stoi(argv[i + 1]);
        }
        else if (strcmp(arg, "-h") == 0)
        {
            usage(argc, argv);
            help();
            exit(1);
        }
        else
        {
            std::cout << "Invalid argument " << argv[i] << std::endl;
            usage(argc, argv);
            exit(1);
        }

        i += 2;
    }
}

void read_control_points(const char *file, std::vector<Eigen::Vector3f> &output)
{
    std::ifstream readFile(file);
    std::string line;

    float x, y, z;

    while (std::getline(readFile, line))
    {
        std::stringstream ss(line);

        ss >> x >> y >> z;

        output.push_back(Eigen::Vector3f(x, y, z));
    }
}

bool compute_files_stats(Options *options, const char *inputFile, Eigen::Vector3f *controlPoints, int nControlPoints, LASFileStats *stats)
{
    LASreadOpener readOpener;
    readOpener.set_file_name(inputFile);

    if (!readOpener.active())
    {
        return false;
    }

    LASreader *reader = readOpener.open();

    if (!reader)
    {
        return false;
    }

    LASheader *header = &reader->header;

    uint32_t numPoints = reader->header.number_of_point_records;
    uint32_t numSamples = numPoints / options->samplingRate;

    Eigen::Vector3f offset(header->x_offset, header->y_offset, header->z_offset);
    Eigen::Vector3f scale(header->x_scale_factor, header->y_scale_factor, header->z_scale_factor);
    Eigen::Vector3f minBound(header->min_x, header->min_y, header->min_z);
    Eigen::Vector3f maxBound(header->max_x, header->max_y, header->max_z);

    std::vector<float> lasPointData;

    for (int i = 0; i < numSamples; ++i)
    {
        reader->seek(i * options->samplingRate);
        reader->read_point();

        float x = reader->point.get_X() * scale.x() + offset.x();
        float y = reader->point.get_Y() * scale.y() + offset.y();
        float z = reader->point.get_Z() * scale.z() + offset.z();

        lasPointData.push_back(x);
        lasPointData.push_back(y);
        lasPointData.push_back(z);
    }

    stats->kdtree.init(3, lasPointData.data(), numSamples);
    std::vector<float> dzs;
    const Eigen::Vector3f up(0, 0, 1);

    for (int i = 0; i < nControlPoints; ++i)
    {
        Eigen::Vector3f *controlPoint = controlPoints + i;

        if (controlPoint->x() < minBound.x() || controlPoint->x() > maxBound.x() ||
            controlPoint->y() < minBound.y() || controlPoint->y() > maxBound.y() ||
            controlPoint->z() < minBound.z() || controlPoint->z() > maxBound.z())
            continue;

        int nearestNbrs[NBR_QUERY_SIZE];
        int nNbrs;
        float point[3] = {controlPoint->x(), controlPoint->y(), controlPoint->z()};

        stats->kdtree.knn(NBR_QUERY_SIZE, &point[0], &nearestNbrs[0], &nNbrs);

        Eigen::Vector3f nbrPoints[NBR_QUERY_SIZE];
        float minXYDist = std::numeric_limits<float>::max();

        for (int j = 0; j < nNbrs; ++j)
        {
            int idx = nearestNbrs[j] * 3;

            nbrPoints[j].x() = lasPointData[idx];
            nbrPoints[j].y() = lasPointData[idx + 1];
            nbrPoints[j].z() = lasPointData[idx + 2];

            float dx = nbrPoints[j].x() - controlPoint->x();
            float dy = nbrPoints[j].y() - controlPoint->y();
            float dist = sqrt(dx * dx + dy * dy);

            if (dist < minXYDist)
            {
                minXYDist = dist;
            }
        }

        if (minXYDist > options->distThresh)
        {
            continue;
        }

        Eigen::Vector3f centroid(0, 0, 0);

        for (int j = 0; j < nNbrs; ++j)
        {
            centroid += nbrPoints[j];
        }

        centroid /= nNbrs;

        for (int j = 0; j < nNbrs; ++j)
        {
            nbrPoints[j] -= centroid;
        }
        float xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;

        for (int j = 0; j < nNbrs; ++j)
        {
            Eigen::Vector3f *point = &nbrPoints[j];

            xx += point->x() * point->x();
            xy += point->x() * point->y();
            xz += point->x() * point->z();
            yy += point->y() * point->y();
            yz += point->y() * point->z();
            zz += point->z() * point->z();
        }

        Eigen::Matrix3f cov;

        cov << xx, xy, xz, xy, yy, yz, xz, yz, zz;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(cov);
        Eigen::Vector3f eigenValues = eigensolver.eigenvalues();
        Eigen::Matrix3f eigenVectors = eigensolver.eigenvectors();

        Eigen::Vector3f bestFitPlaneNormal;
        float minEV = std::numeric_limits<float>::max();

        for (int j = 0; j < 3; ++j)
        {
            if (abs(eigenValues[j]) < minEV)
            {
                minEV = abs(eigenValues[j]);
                bestFitPlaneNormal = eigenVectors.col(j);
            }
        }

        float dotProd = bestFitPlaneNormal.dot(up);
        float angle = acos(dotProd);

        angle = std::min(angle, (float)M_PI - angle);

        if (angle > options->angleThresh)
        {
            continue;
        }

        dzs.push_back(controlPoint->z() - centroid.z());
        stats->controlPointIndices.push_back(i);
    }

    float mdz = 0;

    for (int i = 0; i < dzs.size(); ++i)
    {
        mdz += dzs[i];
    }

    mdz /= dzs.size();

    reader->close();
    delete reader;

    stats->file = std::filesystem::path(inputFile);
    stats->dz = mdz;

    return !std::isnan(mdz);
}

// void computeOverlaps(std::vector<LASFileStats> &fileStats, std::vector<std::vector<int>> &overlapGraph)
// {
//     std::vector<Eigen::Vector3f> mins;
//     std::vector<Eigen::Vector3f> maxs;

//     overlapGraph.resize(fileStats.size());

//     for (int i = 0; i < fileStats.size(); ++i)
//     {
//         for (int j = i + 1; j < fileStats.size(); ++j)
//         {
//             std::vector<int> intersection(std::max(fileStats[i].controlPointIndices.size(), fileStats[j].controlPointIndices.size()));

//             std::vector<int>::iterator it = std::set_intersection(
//                 fileStats[i].controlPointIndices.begin(),
//                 fileStats[i].controlPointIndices.end(),
//                 fileStats[j].controlPointIndices.begin(),
//                 fileStats[j].controlPointIndices.end(),
//                 intersection.begin());

//             intersection.resize(it - intersection.begin());

//             if (intersection.size() >= 10)
//             {
//                 std::cout << "Found overlap "
//                           << "(" << intersection.size() << " points)"
//                           << " between " << fileStats[i].file << " and " << fileStats[j].file << std::endl;

//                 overlapGraph[i].push_back(j);
//                 overlapGraph[j].push_back(i);
//             }
//         }
//     }
// }

bool calibrate(LASFileStats *stats, const char *outputFile)
{
    LASreadOpener readOpener;
    LASwriteOpener writeOpener;

    std::string lasFileStr = stats->file.string();

    readOpener.set_file_name(lasFileStr.c_str());
    writeOpener.set_file_name(outputFile);

    if (!readOpener.active() || !writeOpener.active())
    {
        return false;
    }

    LASreader *reader = readOpener.open();

    if (!reader)
    {
        return false;
    }

    LASwriter *writer = writeOpener.open(&reader->header);

    if (!writer)
    {
        return false;
    }

    float dz = stats->dz / reader->header.z_scale_factor; // unscale dz for writing

    while (reader->read_point())
    {
        LASpoint *point = &reader->point;
        point->set_Z(point->get_Z() + dz);
        writer->write_point(point);
        writer->update_inventory(point);
    }

    writer->update_header(&reader->header, true);

    writer->close();
    delete writer;

    reader->close();
    delete reader;

    return true;
}

struct ComputeStatsParams
{
    std::string inputFile;
    int nControlPoints;
    Eigen::Vector3f *controlPoints;
    Options *options;
    LASFileStats *stats;
    bool status;
};

struct CalibrateParams
{
    std::string outputFile;
    LASFileStats *stats;
    bool status;
};

void run_compute_file_stats(ComputeStatsParams *params)
{
    params->status = compute_files_stats(params->options, params->inputFile.c_str(), params->controlPoints, params->nControlPoints, params->stats);
}

void run_calibrate(CalibrateParams *params)
{
    params->status = calibrate(params->stats, params->outputFile.c_str());
}

void summarize(const char *outputFile, std::vector<LASFileStats> &stats)
{
    std::ofstream outStream(outputFile);

    outStream << "filename,dz" << std::endl;

    for (int i = 0; i < stats.size(); ++i)
    {
        outStream << stats[i].file.filename().string() << "," << stats[i].dz << std::endl;
    }

    outStream.close();
}

int main(int argc, const char *argv[])
{

    Options options;
    parseArgs(argc, argv, &options);

    std::filesystem::path inputDir = options.inputDir;
    std::filesystem::path controlFile = options.controlFile;

    if (options.inputDir.empty() || options.controlFile.empty())
    {
        usage(argc, argv);
        exit(1);
    }

    if (!std::filesystem::exists(inputDir))
    {
        std::cout << "Input directory " << inputDir.string() << " does not exist" << std::endl;
        exit(1);
    }

    if (!std::filesystem::exists(controlFile))
    {
        std::cout << "Control file " << controlFile.string() << " does not exist" << std::endl;
        exit(1);
    }

    std::filesystem::path outputDir = options.outputDir;

    if (outputDir.empty())
    {
        outputDir = inputDir / std::filesystem::path("calibrated");
    }

    if (!std::filesystem::exists(outputDir))
    {
        bool created = std::filesystem::create_directory(outputDir);

        if (!created)
        {
            std::cout << "Failed to create directory " << outputDir.string() << "\n";
            exit(1);
        }
        else
        {
            std::cout << "Created output directory " << outputDir.string() << "\n";
        }
    }
    else
    {
        std::cout << "Output directory " << outputDir.string() << " already exists"
                  << "\n";
    }

    std::filesystem::path logFile = outputDir / std::filesystem::path("calibrate.log.txt");
    std::filesystem::path summaryFile = outputDir / std::filesystem::path("calibrate.summary.csv");

    std::string logFileStr = logFile.string();
    std::string summaryFileStr = summaryFile.string();

    Logger logger(logFileStr.c_str(), false);

    logger << "Sampling rate: " << options.samplingRate << "\n";
    logger << "Angle threshold: " << (int)round(options.angleThresh * 180 / M_PI) << "deg"
           << "\n";
    logger << "Distance threshold: " << options.distThresh << "ft"
           << "\n";
    logger << "Using " << options.maxThreads << " threads"
           << "\n";

    std::vector<std::filesystem::path> inputFiles;

    for (const std::filesystem::directory_entry &entry : std::filesystem::directory_iterator(inputDir))
    {
        std::filesystem::path file = entry.path();

        if (file.has_extension())
        {
            std::filesystem::path ext = file.extension();

            if (ext == ".laz" || ext == ".las")
            {
                inputFiles.push_back(file.filename());
            }
        }
    }

    logger << "Found " << inputFiles.size() << " input files:"
           << "\n";

    for (int i = 0; i < inputFiles.size(); ++i)
    {
        logger << "\t" << inputFiles[i].string() << "\n";
    }

    std::vector<Eigen::Vector3f> controlPoints;
    std::string controlFileStr = controlFile.string();
    read_control_points(controlFileStr.c_str(), controlPoints);

    logger << "Processed control points from " << controlFileStr << "\n";

    std::vector<LASFileStats> fileStats(inputFiles.size());
    std::vector<ComputeStatsParams> computeStatsParams(inputFiles.size());
    std::vector<CalibrateParams> calibrateParams(inputFiles.size());
    // std::vector<std::vector<int>> overlapGraph;

    logger << "Computing calibration stats"
           << "\n";

    std::vector<std::thread *> threads(inputFiles.size());

    for (int i = 0; i < fileStats.size(); ++i)
    {
        std::filesystem::path inputFile = inputDir / inputFiles[i];
        std::string outputFileName;
        outputFileName += inputFiles[i].stem().string();
        outputFileName += ".calibrated";
        outputFileName += inputFiles[i].extension().string();
        std::filesystem::path outputFile = outputDir / std::filesystem::path(outputFileName);

        ComputeStatsParams *computeStatsParam = computeStatsParams.data() + i;
        computeStatsParam->options = &options;
        computeStatsParam->inputFile = inputFile.string();
        computeStatsParam->controlPoints = controlPoints.data();
        computeStatsParam->nControlPoints = controlPoints.size();
        computeStatsParam->stats = fileStats.data() + i;

        CalibrateParams *calibrateParam = calibrateParams.data() + i;
        calibrateParam->stats = fileStats.data() + i;
        calibrateParam->outputFile = outputFile.string();
    }

    int ti;

    ti = 0;

    while (ti < fileStats.size())
    {
        int ei = std::min(ti + options.maxThreads, (int)fileStats.size());

        for (int i = ti; i < ei; ++i)
        {
            threads[i] = new std::thread(run_compute_file_stats, computeStatsParams.data() + i);
        }

        for (int i = ti; i < ei; ++i)
        {
            threads[i]->join();
            delete threads[i];

            logger << "\t";

            if (computeStatsParams[i].stats->controlPointIndices.size() == 0) {
                logger << "No control points found for";
            }
            else {
                if (!computeStatsParams[i].status)
                {
                    logger << "Failed to compute stats for";
                }
                else
                {
                    logger << "Computed stats for";
                }
            }

            logger << " " << inputFiles[i].string() << " (" << i + 1 << "/" << fileStats.size() << ")"
                   << "\n";
        }

        ti = ei;
    }

    logger << "Computed calibration values:"
           << "\n";

    for (int i = 0; i < fileStats.size(); ++i)
    {
        logger << "\t" << std::left
               << std::setw(30) << inputFiles[i].string()
               << std::setw(10) << fileStats[i].dz << "\n";
    }

    logger << "Applying calibration values"
           << "\n";

    ti = 0;

    while (ti < fileStats.size())
    {
        int ei = std::min(ti + options.maxThreads, (int)fileStats.size());

        for (int i = ti; i < ei; ++i)
        {
            if (std::isnan(fileStats[i].dz))
            {
                threads[i] = nullptr;
            }
            else
            {
                threads[i] = new std::thread(run_calibrate, calibrateParams.data() + i);
            }
        }

        for (int i = ti; i < ei; ++i)
        {
            bool skipped = threads[i] == nullptr;

            if (threads[i])
            {
                threads[i]->join();
                delete threads[i];
            }

            logger << "\t";

            if (skipped)
            {
                logger << "No control points found. Skipped";
            }
            else
            {
                if (!calibrateParams[i].status)
                {
                    logger << "Failed to apply calibration to";
                }
                else
                {
                    logger << "Applied calibration to";
                }
            }

            logger << " " << calibrateParams[i].outputFile << " (" << i + 1 << "/" << fileStats.size() << ")"
                   << "\n";
        }

        ti = ei;
    }

    logger << "Finished"
           << "\n";

    logger.close();

    std::cout << "Check " << logFileStr << " for logs" << std::endl;

    summarize(summaryFileStr.c_str(), fileStats);

    std::cout << "Check " << summaryFileStr << " for summary stats" << std::endl;

    // computeOverlaps(fileStats, overlapGraph);

    // std::vector<std::vector<int>> overlapGroups;
    // std::vector<bool> visited(overlapGraph.size());

    // std::fill(visited.begin(), visited.end(), false);

    // for (int i = 0; i < visited.size(); ++i)
    // {
    //     if (!visited[i])
    //     {
    //         std::vector<int> group;
    //         std::vector<int> stack;

    //         visited[i] = true;
    //         stack.push_back(i);

    //         while (stack.size())
    //         {
    //             int current = stack.back();
    //             stack.pop_back();
    //             group.push_back(current);

    //             std::vector<int> &nbrs = overlapGraph[current];

    //             for (int j = 0; j < nbrs.size(); ++j)
    //             {
    //                 if (!visited[nbrs[j]])
    //                 {
    //                     visited[nbrs[j]] = true;
    //                     stack.push_back(nbrs[j]);
    //                 }
    //             }
    //         }

    //         overlapGroups.push_back(group);
    //     }
    // }

    // for (int i = 0; i < overlapGroups.size(); ++i)
    // {
    //     logger << "Group " << i + 1 << ":" << "\n";

    //     for (int j = 0; j < overlapGroups[i].size(); ++j)
    //     {
    //         logger << "\t" << fileStats[overlapGroups[i][j]].file << "\n";
    //     }
    // }

    // logger << "\n";

    // for (int i = 0; i < fileStats.size(); ++i)
    // {
    //     logger << inputFiles[i] << "\n";
    //     logger << "Found control points: ";

    //     for (int j = 0; j < fileStats[i].controlPointIndices.size(); ++j)
    //     {
    //         logger << fileStats[i].controlPointIndices[j];

    //         if (j < fileStats[i].controlPointIndices.size() - 1)
    //         {
    //             logger << ", ";
    //         }
    //     }

    //     logger << "\n";
    //     logger << "Dz: " << fileStats[i].dz << "\n";
    // }

    return 0;
}
