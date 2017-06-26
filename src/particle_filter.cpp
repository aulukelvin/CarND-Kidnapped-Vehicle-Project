/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "helper_functions.h"
#include "particle_filter.h"

using namespace std;
static default_random_engine generator;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 400;

    std::normal_distribution<double> distributionX(0, std[0]);
    std::normal_distribution<double> distributionY(0, std[1]);
    std::normal_distribution<double> distributionTheta(0, std[2]);

    particles = std::vector<Particle> (num_particles);
    for (int i = 0; i < num_particles; i++) {
        Particle ptc;
        ptc.x = x + distributionX(generator);
        ptc.y = y + distributionY(generator);
        ptc.theta = Normalize(theta + distributionTheta(generator));
        particles[i] = ptc;
    }

    cout << "init: measure: " << x << " - " << y << " - " << theta << endl;
    cout << "init: result " << particles[0].x << " - " << particles[0].y << " - " << particles[0].theta << endl;
    is_initialized = true;
    cout << "initialization done" << endl;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // Add measurements to each particle and add random Gaussian noise.
    std::normal_distribution<double> distributionX(0, std_pos[0]);
    std::normal_distribution<double> distributionY(0, std_pos[1]);
    std::normal_distribution<double> distributionTheta(0, std_pos[2]);

    yaw_rate = Normalize(yaw_rate);
    cout << "predict: prior " << particles[0].x << " - " << particles[0].y << " - " << particles[0].theta << " - " << velocity << " - " << yaw_rate << endl;
    for (int i = 0; i < num_particles; i++) {
        Particle ptc = particles[i];
        if (fabs(yaw_rate) < 0.00001) {
            ptc.x += velocity * delta_t * cos(ptc.theta) + distributionX(generator);
            ptc.y += velocity * delta_t * sin(ptc.theta) + distributionY(generator);
            ptc.theta = Normalize(ptc.theta + distributionTheta(generator));
        } else {
            ptc.x += velocity / yaw_rate * (sin(ptc.theta + yaw_rate * delta_t) - sin(ptc.theta)) + distributionX(generator);
            ptc.y += velocity / yaw_rate * (cos(ptc.theta) - cos(ptc.theta + yaw_rate * delta_t)) + distributionY(generator);
            ptc.theta = Normalize(ptc.theta + yaw_rate * delta_t + distributionTheta(generator));
        }
        particles[i] = ptc;
    }
    cout << "predict: posterior " << particles[0].x << " - " << particles[0].y << " - " << particles[0].theta << endl;

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    double min_dist;
    int idx;

    double dist_tmp;
    for (int i = 0; i < predicted.size(); i++) {
        min_dist = 99999.0;
        idx = -1;

        LandmarkObs pred_L = predicted[i];

        for (int j = 0; j < observations.size(); j++) {
            LandmarkObs obs_L = observations[j];
            dist_tmp = dist(pred_L.x, pred_L.y, obs_L.x, obs_L.y);
            if (dist_tmp < min_dist) {
                min_dist = dist_tmp;
                idx = j;
            }
        }
        pred_L.id = idx;

        predicted[i] = pred_L;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
        std::vector<LandmarkObs> observations, Map map_landmarks) {
    // Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    double denominator = (2 * M_PI * std_landmark[0] * std_landmark[1]);

    for (int i = 0; i < num_particles; i++) {
        Particle ptc = particles[i];
        LandmarkObs tmp;
        double x = ptc.x;
        double y = ptc.y;
        double theta = ptc.theta;
        double weight = 1.0;

        // transformation
        vector<LandmarkObs> predictions = std::vector<LandmarkObs> (observations.size());
        for (int j = 0; j < observations.size(); j++) {
            LandmarkObs obs = observations[j];
            tmp = LandmarkObs();

            tmp.x = x * cos(theta) - y * sin(theta) + obs.x;
            tmp.y = x * sin(theta) + y * cos(theta) + obs.y;
            tmp.id = -1;
            predictions[j] = tmp;
        }

        // associate
        dataAssociation(predictions, observations);

        // calculate weight
        for (int j = 0; j < predictions.size(); j++) {
            LandmarkObs pred = predictions[j];
            weight *= exp(-(pow(pred.x - map_landmarks.landmark_list[pred.id].x_f, 2) / (2 * std_landmark[0] * std_landmark[0])
                    + pow(pred.y - map_landmarks.landmark_list[pred.id].y_f, 2) / (2 * std_landmark[1] * std_landmark[1])))
                    / denominator;
        }
        ptc.weight = weight;
        particles[i] = ptc;
    }

    cout << "weighttt: " << particles[0].weight << endl;
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    vector<Particle> new_particles;

    // get all of the current weights
    vector<double> weights;
    for (int i = 0; i < num_particles; i++) {
        weights.push_back(particles[i].weight);
    }

    // generate random starting index for resampling wheel
    uniform_int_distribution<int> uniintdist(0, num_particles - 1);
    auto index = uniintdist(generator);

    // get max weight
    double max_weight = *max_element(weights.begin(), weights.end());

    // uniform random distribution [0.0, max_weight)
    uniform_real_distribution<double> unirealdist(0.0, max_weight);

    double beta = 0.0;

    // spin the resample wheel!
    for (int i = 0; i < num_particles; i++) {
        beta += unirealdist(generator) * 2.0;
        while (beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    }

    particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1); // get rid of the trailing space
    return s;
}
