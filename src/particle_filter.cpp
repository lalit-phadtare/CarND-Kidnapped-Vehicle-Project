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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	//LP: 
	//initiate num. of particles
	this->num_particles = 100;

	// random number generator
	default_random_engine gen;

	// This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < this->num_particles; ++i) {

		//push back a Particle structure on the member vector and initialize it using the distribution. 
		Particle temp;
		this->particles.push_back(temp);
		this->particles.back().id = i;
		this->particles.back().x = dist_x(gen);
		this->particles.back().y = dist_y(gen);
		this->particles.back().theta = dist_theta(gen);
		this->particles.back().weight = 1.0;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	std::default_random_engine gen;

	for (int i = 0; i < this->num_particles; i++) {
		// calculate new state
		if (fabs(yaw_rate) < 0.00001) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else
		{
			this->particles[i].x = this->particles[i].x + ((velocity / yaw_rate)*(sin(this->particles[i].theta + (delta_t*yaw_rate)) - sin(this->particles[i].theta)));
			this->particles[i].y = this->particles[i].y + ((velocity / yaw_rate)*(cos(this->particles[i].theta) - cos(this->particles[i].theta + (delta_t*yaw_rate))));
		}
			this->particles[i].theta = this->particles[i].theta + (delta_t*yaw_rate);
		
		//normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		//normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		//normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

		normal_distribution<double> dist_x(0, std_pos[0]);
		normal_distribution<double> dist_y(0, std_pos[1]);
		normal_distribution<double> dist_theta(0, std_pos[2]);

		this->particles[i].x = this->particles[i].x + dist_x(gen);
		this->particles[i].y = this->particles[i].y + dist_y(gen);
		this->particles[i].theta = this->particles[i].theta + dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	for (unsigned int i = 0; i < observations.size(); i++) {
		double min_dist = std::numeric_limits<double>::max();
		LandmarkObs curr_obs = observations[i];
		for (unsigned int j = 0; j < predicted.size(); j++) {
			double curr_dist = dist(predicted[j].x, predicted[j].y, curr_obs.x, curr_obs.y);
			if (curr_dist < min_dist) {
				min_dist = curr_dist;
				observations[i].id = predicted[j].id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// write to before update
	//this->write("stage1");

	for (int i = 0; i < this->num_particles; i++) {
		//convert observations from car coords to map coords 
		std::vector<LandmarkObs> transformedObs = ParticleFilter::transformObs(this->particles[i], observations);

		//get all the map landmarks within the sensor range of the particle's current position
		std::vector<LandmarkObs> predictedObs = ParticleFilter::predictObs(this->particles[i], sensor_range, map_landmarks);

		//call the data association to get transformed obs. coords match to actual landmarks
		ParticleFilter::dataAssociation(predictedObs, transformedObs);

		//calculate measurement probability for every transformed obs.
		//calculate final weight of the paticle
		double sig_x = std_landmark[0];
		double sig_y = std_landmark[1];
		this->particles[i].weight = 1.0;
		for (unsigned int j = 0; j < transformedObs.size(); j++) {
			for (unsigned int k = 0; k < predictedObs.size(); k++) {
				if (transformedObs[j].id == predictedObs[k].id) {
					double x_obs = transformedObs[j].x;
					double y_obs = transformedObs[j].y;
					double mu_x = predictedObs[k].x;
					double mu_y = predictedObs[k].y;

					// calculate normalization term
					double gauss_norm = (1 / (2 * M_PI * sig_x * sig_y));

					// calculate exponent
					double exponent = (pow((x_obs - mu_x), 2)) / (2 * pow(sig_x, 2)) + (pow((y_obs - mu_y), 2)) / (2 * pow(sig_y, 2));

					// calculate weight using normalization terms and exponent
					double weight = gauss_norm * exp(-exponent);

					//update weight of the current particle
					this->particles[i].weight *= weight;

				}
			}
		}

		// write to before update
		//this->write("stage2");
	}
}

void ParticleFilter::resample() {
	std::vector<Particle> resampledParticles;
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution_int(0, this->num_particles - 1);
	std::uniform_real_distribution<> distribution_real(0, 1);
	auto index = distribution_int(generator);
	double beta = 0.0;
	double mw = this->getMaxWeight();
	for (int i = 0; i < this->num_particles; i++) {
		beta += distribution_real(generator) * 2.0 * mw;
		while (beta > this->particles[index].weight) {
			beta -= this->particles[index].weight;
			index = (index + 1) % this->num_particles;
		}
		resampledParticles.push_back(this->particles[index]);
	}
	this->particles = resampledParticles;

}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
	const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}

std::vector<LandmarkObs> ParticleFilter::transformObs(Particle part, std::vector<LandmarkObs> obs) {
	std::vector<LandmarkObs> transformedObs;
	for (unsigned int i = 0; i < obs.size(); i++) {
		LandmarkObs obs_t;

		// transform to map x coordinate
		obs_t.x = part.x + (cos(part.theta) * obs[i].x) - (sin(part.theta) * obs[i].y);

		// transform to map y coordinate
		obs_t.y = part.y + (sin(part.theta) * obs[i].x) + (cos(part.theta) * obs[i].y);

		//copy the id
		obs_t.id = part.id;

		transformedObs.push_back(obs_t);
	}
	return transformedObs;
}

std::vector<LandmarkObs> ParticleFilter::predictObs(Particle part, double sensor_range, const Map &map_landmarks) {
	std::vector<LandmarkObs> predictedObs;
	for (unsigned int i = 0; i < map_landmarks.landmark_list.size(); i++) {
		if ((fabs(map_landmarks.landmark_list[i].x_f - part.x) < 50) && (fabs(map_landmarks.landmark_list[i].y_f - part.y) < 50)) {
			LandmarkObs temp;
			temp.x = map_landmarks.landmark_list[i].x_f;
			temp.y = map_landmarks.landmark_list[i].y_f;
			temp.id = map_landmarks.landmark_list[i].id_i;
			predictedObs.push_back(temp);
		}
	}
	return predictedObs;
}

double ParticleFilter::getMaxWeight() {
	double max_weight = 0;
	for (int i = 0; i < this->num_particles; i++) {
		if (this->particles[i].weight > max_weight) {
			max_weight = this->particles[i].weight;
		}
	}
	return max_weight;
}

//void ParticleFilter::write(std::string stage) {
//	// You don't need to modify this file.
//	std::ofstream dataFile;
//	dataFile.open("data.txt", ios::app);
//	dataFile << stage << " " << std::endl;
//	for (int i = 0; i < num_particles; ++i) {
//		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << " " << particles[i].weight << std::endl;
//	}
//	dataFile.close();
//}