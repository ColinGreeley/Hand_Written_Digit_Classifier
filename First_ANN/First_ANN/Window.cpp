#include "Window.h"
#include <string>

// for testing only!! //
void renderWindow() {
	
	int s = 0;
	std::string guessString;
	std::string accuracyString;
	ANN a;
	sf::RectangleShape pixel;
	sf::CircleShape neuron;
	sf::VertexArray neuralConnection(sf::Lines, 2);
	sf::RectangleShape r;

	pixel.setSize(sf::Vector2f(12, 12));
	pixel.setPosition(60, 600);
	neuron.setRadius(10);
	neuron.setPosition(500, 500);
	neuron.setFillColor(sf::Color::Black);
	neuron.setOutlineColor(sf::Color::White);
	neuron.setOutlineThickness(1);
	neuron.setOrigin(neuron.getRadius() / 2, neuron.getRadius() / 2);
	r.setSize(sf::Vector2f(pixel.getSize().x * 28, pixel.getSize().y * 28));
	r.setPosition(pixel.getPosition());
	r.setFillColor(sf::Color(100, 100, 100));
	r.setOutlineColor(sf::Color::Black);
	r.setOutlineThickness(5);

	a.loadNetworkValues();
	//a.generateRadomWeights();
	//a.generateRandomBias();
	a.createTestImages();

	sf::Text guess;
	sf::Text accuracy;
	sf::Font guessFont;

	guessFont.loadFromFile("Fonts/pixelfont.ttf");
	guess.setFillColor(sf::Color::White);
	guess.setPosition(sf::Vector2f(500, 300));
	guess.setCharacterSize(45);
	guess.setFont(guessFont);
	accuracy.setFillColor(sf::Color::White);
	accuracy.setPosition(sf::Vector2f(500, 200));
	accuracy.setCharacterSize(30);
	accuracy.setFont(guessFont);
	
	
	sf::RenderWindow window(sf::VideoMode(1920, 1080), "First Artificial Neural Network", sf::Style::Fullscreen);

	while (window.isOpen()) {

		sf::Event evnt;
		while (window.pollEvent(evnt)) {

			switch (evnt.type) {
			case sf::Event::Closed:
				window.close();
				break;
			case sf::Event::KeyPressed:
				if (sf::Keyboard::Escape) {
					window.close();
					break;
				}
			}
		test(a, pixel, r, neuron, neuralConnection, window, guess, accuracy, guessString, accuracyString);
		break;
		}
		break;
	}
	a.freeMemory();
	a.imageDataSet.freeMemory();
}

void test(ANN a, sf::RectangleShape p, sf::RectangleShape r, sf::CircleShape n, sf::VertexArray nc, sf::RenderWindow &w, sf::Text g, sf::Text acc, std::string gs, std::string as) {
	
	int s = 0;
	bool correctAnswer;

	for (int i = 0; i < TESTSET_SIZE; i++) {
		w.clear(sf::Color::Color(50, 50, 50));
		gs.clear();
		a.setImageLable(a.imageDataSet.label[i]);
		for (int j = 0; j < INPUT_SIZE; j++) {
			a.setInputLayer(a.imageDataSet.pixleMap[i][j], j);
		}
		a.feedForward();
		correctAnswer = a.findCost();
		//a.backpropagation(1);
		gs.append("Computer's guess: ");
		gs.append(std::to_string(a.computersChoice()));
		g.setString(gs);
		w.draw(g);
		if (correctAnswer == true)
			r.setFillColor(sf::Color(100, 100, 100));
		else
			r.setFillColor(sf::Color::Red);
		w.draw(r);
		drawImage(a, p, w, i);
		drawNeuralNetwork(a, n, nc, w, acc, as);
		s++;
		as.clear();
		as.append(std::to_string((int)(((float)a.getCorrectAnswers() / (float)s) * 100 * 1000.0) / 1000));
		as.append("% accuracy");
		acc.setString(as);
		w.draw(acc);
		w.display();
		while (!sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space)) { if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape)) break; }
		while (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space)) { if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape)) break; }
	}
	
}

void drawImage(ANN a, sf::RectangleShape p, sf::RenderWindow &w, int index) {

	int count = 0;
	int yPosition = 0;
	int xCounter = 0;

	for (int i = 0; i < 784; i++) {
		p.setPosition(p.getPosition().x + p.getSize().x - xCounter, p.getPosition().y + yPosition);
		p.setFillColor(sf::Color(255, 255, 255, a.imageDataSet.pixleMap[index][i] * 255));
		w.draw(p);
		count++;
		if (count % 28 == 0) {
			count = 0;
			yPosition = p.getSize().y;
			xCounter = 28 * p.getSize().x;
		}
		else {
			yPosition = 0;
			xCounter = 0;
		}
		//std::cout << a.imageDataSet.pixleMap[index][i] << "\n";
	}
}

void drawNeuralNetwork(ANN a, sf::CircleShape n, sf::VertexArray nc, sf::RenderWindow &w, sf::Text acc, std::string as) {

	int yCounter = 25;
	
	nc[0].position = sf::Vector2f(n.getPosition().x, n.getPosition().y + 20);
	nc[0].color = sf::Color::White;
	nc[1].position = sf::Vector2f(700, 555);
	for (int i = 0; i < 20; i++) {
		for (int j = 0; j < 16; j++) {
			nc[1].position = sf::Vector2f(nc[0].position.x + 200, nc[1].position.y + yCounter);
			if (a.getInputWeights(j, i) > 0) nc[0].color = nc[1].color = sf::Color(255, 0, 0, a.getInputWeights(j, i) * 255);
			else nc[0].color = nc[1].color = sf::Color(0, 0, 255, a.getInputWeights(j, i) * 255);
			w.draw(nc);
		}
		 nc[0].position = sf::Vector2f(nc[0].position.x, nc[0].position.y + yCounter + 1);
		 nc[1].position = sf::Vector2f(nc[0].position.x + 200, 555);
	}

	nc[0].position = sf::Vector2f(700, 575);
	nc[0].color = sf::Color::White;
	nc[1].position = sf::Vector2f(900, 555);
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			nc[1].position = sf::Vector2f(nc[0].position.x + 200, nc[1].position.y + yCounter);
			if (a.getHiddenLayer1Weights(j, i) > 0) nc[0].color = nc[1].color = sf::Color(255, 0, 0, a.getHiddenLayer1Weights(j, i) * 255);
			else nc[0].color = nc[1].color = sf::Color(0, 0, 255, a.getHiddenLayer1Weights(j, i) * 255);
			w.draw(nc);
		}
		nc[0].position = sf::Vector2f(nc[0].position.x, nc[0].position.y + yCounter + 1);
		nc[1].position = sf::Vector2f(nc[0].position.x + 200, 555);
	}

	nc[0].position = sf::Vector2f(900, 575);
	nc[0].color = sf::Color::White;
	nc[1].position = sf::Vector2f(1100, 645);
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 10; j++) {
			nc[1].position = sf::Vector2f(nc[0].position.x + 200, nc[1].position.y + yCounter);
			if (a.getHiddenLayer2Weights(j, i) > 0) nc[0].color = nc[1].color = sf::Color(255, 0, 0, a.getHiddenLayer2Weights(j, i) * 255);
			else nc[0].color = nc[1].color = sf::Color(0, 0, 255, a.getHiddenLayer2Weights(j, i) * 255);
			w.draw(nc);
		}
		nc[0].position = sf::Vector2f(nc[0].position.x, nc[0].position.y + yCounter + 1);
		nc[1].position = sf::Vector2f(nc[0].position.x + 200, 645);
	}


	for (int i = 0; i < 20; i++) {
		n.setPosition(n.getPosition().x, n.getPosition().y + yCounter);
		n.setFillColor(sf::Color::Blue);
		w.draw(n);
	}

	n.setPosition(700, 555);
	for (int i = 0; i < 16; i++) {
		n.setPosition(n.getPosition().x, n.getPosition().y + yCounter);
		n.setFillColor(sf::Color(a.getHiddenLayer1(i) * 255, 0, 0));
		w.draw(n);
	}

	n.setPosition(900, 555);
	for (int i = 0; i < 16; i++) {
		n.setPosition(n.getPosition().x, n.getPosition().y + yCounter);
		n.setFillColor(sf::Color(a.getHiddenLayer2(i) * 255, 0, 0));
		w.draw(n);
	}

	n.setPosition(1100, 640);
	for (int i = 0; i < 10; i++) {
		n.setPosition(n.getPosition().x, n.getPosition().y + yCounter);
		n.setFillColor(sf::Color(a.getOutputLayer(i) * 255, 0, 0));
		w.draw(n);
	}

	acc.setPosition(1130, 645);
	acc.setCharacterSize(10);
	for (int i = 0; i < 10; i++) {
		as.clear();
		as.append(std::to_string(i));
		acc.setString(as);
		acc.setPosition(acc.getPosition().x, acc.getPosition().y + 25);
		w.draw(acc);
	}

	nc[0].color = sf::Color::White;
	nc[1].color = sf::Color::White;
	nc[0].position = sf::Vector2f(410, 670);
	nc[1].position = sf::Vector2f(460, 670);
	w.draw(nc);
	nc[0].position = sf::Vector2f(460, 670);
	nc[1].position = sf::Vector2f(440, 650);
	w.draw(nc);
	nc[0].position = sf::Vector2f(460, 670);
	nc[1].position = sf::Vector2f(440, 690);
	w.draw(nc);

	nc[0].position = sf::Vector2f(410, 770);
	nc[1].position = sf::Vector2f(460, 770);
	w.draw(nc);
	nc[0].position = sf::Vector2f(460, 770);
	nc[1].position = sf::Vector2f(440, 750);
	w.draw(nc);
	nc[0].position = sf::Vector2f(460, 770);
	nc[1].position = sf::Vector2f(440, 790);
	w.draw(nc);

	nc[0].position = sf::Vector2f(410, 870);
	nc[1].position = sf::Vector2f(460, 870);
	w.draw(nc);
	nc[0].position = sf::Vector2f(460, 870);
	nc[1].position = sf::Vector2f(440, 850);
	w.draw(nc);
	nc[0].position = sf::Vector2f(460, 870);
	nc[1].position = sf::Vector2f(440, 890);
	w.draw(nc);
}