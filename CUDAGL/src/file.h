#pragma once
#include <string>
#include <fstream>
#include <iostream>

inline std::string readFile(const std::string &path)
{
	std::ifstream in(path);
	if (!in.good()) {
		std::cerr << path << " not good!" << std::endl;
	}
	std::string contents((std::istreambuf_iterator<char>(in)),
		std::istreambuf_iterator<char>());
	in.close();
	return contents;
}
