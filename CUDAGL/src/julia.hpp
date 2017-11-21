#pragma once
#include <GLEW/glew.h>
#include <GLFW/glfw3.h>
#include "program.hpp"

class Julia : public Program
{
	
public:

	~Julia() override;

	void onInit(int width, int height, int argc, char** argv) override;
	void onResize(int width, int height) override;
	void onLoop() override;
	void onClose() override;
	void onKey(int key, int scancode, int action, int mods) override;
	void onMouseMove(double xpos, double ypos) override;
	void onMouseButton(int button, int action, int mods) override;

};

