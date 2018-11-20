#pragma once
#include "ANN.h"
#include <SFML\Graphics.hpp>

void renderWindow();
void test(ANN a, sf::RectangleShape p, sf::RectangleShape r, sf::CircleShape n, sf::VertexArray nc, sf::RenderWindow &w, sf::Text g, sf::Text acc, std::string gs, std::string as);
void drawImage(ANN a, sf::RectangleShape p, sf::RenderWindow &w, int index);
void drawNeuralNetwork(ANN a, sf::CircleShape n, sf::VertexArray nc, sf::RenderWindow &w, sf::Text acc, std::string as);