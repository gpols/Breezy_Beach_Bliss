#pragma once

#include "ofMain.h"
#include "PRamp.h"

// Struct to hold terrain information (height and gradient)
struct terrainNumbers {
    float height;
    glm::vec2 gradient;
};

// Struct to hold transform information for each tree
struct treeTransform {
    glm::vec3 position;
    glm::vec3 scale;
    float rotation; // Add any other transform parameters you need
};

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
        void resetAnimations();
    // Function to calculate terrain information at a given position
    terrainNumbers calculateTerrain(glm::vec2 pos);

    
		ofEasyCam cam;
		ofShader shader;
        ofMesh sphereMesh;
        ofMesh torusMesh;
        ofMesh planeMesh;
        ofMesh coneMesh;
        ofMesh normalsMesh;
        ofImage sphereTex;
        ofImage torusTex;
        ofImage coneTex;
        ofImage cylinderTex;
        ofTexture tex;
        ofShader shaderSprites;
        ofMesh spriteMesh;
        ofMesh cylinderMesh;

    
    // to add animations
    PRamp spherePosX, spherePosY, torusPosX, torusPosY, cloudsPos;
    
    bool startAnimation; 
    float scale;
    bool wind;
    bool cyclone;
    int sphereRotation;
    int torusRotation;
    glm::vec3 spherePosition;
    glm::vec3 torusPosition;
    glm::vec3 cloudsPosition;
    glm::vec3 conePosition;
    glm::vec3 cylinderPosition;
    
    ofTrueTypeFont font;
    int fontSize = 24;


     
};
