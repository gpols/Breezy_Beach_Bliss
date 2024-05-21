#include "ofApp.h"
#include "ofGraphicsUtil.h"
using namespace glm;

/*--------------------------------------------------------------
  FUNCTION FOR SPRITES
 // Add a rectangle lying on the xy plane centered around the given position
---------------------------------------------------------------*/
static void addRect(ofMesh & m, glm::vec3 pos, float w = 2.f, float h = 2.f){
    float w_2 = w *0.5;
    float h_2 = h *0.5;
    int Nv = m.getVertices ().size() ;
    m.setMode(OF_PRIMITIVE_TRIANGLES);
    m.addVertex ( pos + vec3 (- w_2 , -h_2 , 0.) );
    m.addVertex ( pos + vec3 ( w_2 , -h_2 , 0.) );
    m.addVertex ( pos + vec3 ( w_2 , h_2 , 0.) );
    m.addVertex ( pos + vec3 (- w_2 , h_2 , 0.) );
    m.addTriangle ( Nv +0 , Nv +1 , Nv +2) ;
    m.addTriangle ( Nv +0 , Nv +2 , Nv +3) ;
}

/*--------------------------------------------------------------
  FUNCTION TO ADD NORMALS
---------------------------------------------------------------*/
void addNormalLines(ofMesh &dst, const ofMesh &src, float len = 0.1, ofFloatColor col = ofFloatColor(1, 0, 0)) {
    if (!src.getNormals().size()) return;
    dst.setMode(OF_PRIMITIVE_LINES);
    for (int i = 0; i < src.getVertices().size(); ++i) {
        auto p = src.getVertices()[i];
        auto N = src.getNormals()[i];
        dst.addVertex(p);
        dst.addVertex(p + N * len);
        for (int k = 0; k < 2; ++k) dst.addColor(col);
    }
}
/*--------------------------------------------------------------
  LIGHTING STRUCTURE
---------------------------------------------------------------*/
static std :: string glslLighting (){
    return R"(
    struct Light {
                // Define struct for representing a light source
                vec3 pos;
                float strength;
                float halfDist;
                float ambient;
                vec3 diffuse;
                vec3 specular;
                };

    struct Material {
                // Define struct for representing material properties
                vec3 diffuse;
                vec3 specular;
                float shine;
                    };
    
    struct LightFall {
                // Define struct for representing light falloff
                vec3 diffuse;
                vec3 specular;
                    };

    // In - place addition : a += b
    void addTo (inout LightFall a , in LightFall b){
    a.diffuse += b.diffuse ;
    a.specular += b.specular ;
    }

    // Compute light components falling on surface
    LightFall computeLightFall ( vec3 pos , vec3 N , vec3 eye , in Light lt , in Material mt ){
    vec3 lightDist = lt.pos - pos ;
    float hh = lt.halfDist * lt.halfDist ;
    float atten = lt.strength * hh /( hh + dot ( lightDist , lightDist ));
    vec3 L = normalize ( lightDist );

    // diffuse
    float d = max(dot (N , L) , 0.);
    d += lt.ambient ;

    // specular
    vec3 V = normalize ( eye - pos );
    vec3 H = normalize (L + V);
    float s = pow ( max ( dot (N , H) , 0.), mt.shine );
    LightFall fall ;
    fall.diffuse = lt.diffuse *( d* atten );
    fall.specular = lt.specular *( s* atten );
    return fall ;
    }

    // Get final color reflected off material
    vec3 lightColor (in LightFall f , in Material mt ){
    return f.diffuse * mt .diffuse + f.specular * mt .specular;
    }

)";
}

//--------------------------------------------------------------
void ofApp::setup(){
    
    ofBackground(29, 149, 222);// blue sky

    /*--------------------------------------------------------------
     3D CAMERA SET UP
     ---------------------------------------------------------------*/
    // Setup camera (for 3D rendering)
    cam.setPosition(vec3(0., 0.6, 3.));
    cam.setNearClip(0.05);
    cam.setFarClip(100.);
    ofEnableDepthTest(); // test fragments against depth buffer (for opaque geometry)
    ofEnableNormalizedTexCoords(); // use [0,1] range for UVs
    ofDisableArbTex();
    ofSetFrameRate(40); // must be set for ofGetTargetFrameRate to work
 
    /*--------------------------------------------------------------
     INITIALIZATION
     ---------------------------------------------------------------*/
    scale = 2.;// scaler factor that contribues to animation
    
    //initial position for mesh objects
    spherePosition = glm::vec3(-1., -0.05, 0);
    torusPosition = glm::vec3(-1.5, 0., -0.3);
    conePosition = glm::vec3(0.09, 0.05, -0.9);
    cylinderPosition = glm::vec3(0.4, 1.15, -0.35);
    
    // special font for instructions
    font.load("Summery.ttf", fontSize);

    /*--------------------------------------------------------------
     NOISE TEXTURE
     ---------------------------------------------------------------*/
    {
        int W = 512, H = W;
        auto format = GL_LUMINANCE;
        ofPixels pix; // 2D array of unsigned char
        pix.allocate(W,H, OF_PIXELS_GRAY);
        
        for(int j=0; j<H; ++j){
            for(int i=0; i<W; ++i){
                auto uv = vec2(i,j)/(vec2(W,H)-1.f);
                float val = 0.;
                int octaves = 6;
                float f = 2.; // starting frequency
                float Asum = 0.; // used to scale final amplitude into [0,1]
                for(int k=1; k<=octaves; ++k){
                    float A = 1./f;
                    val += A * ofNoise(uv*f);
                    f *= 2.f;
                    Asum += A;
                }
                pix[j*W + i] = val/Asum*255;
            }}
        tex.allocate(pix);
    }
    
    /*--------------------------------------------------------------
     SPRITE SHADER
     ---------------------------------------------------------------*/
    build(shaderSprites, R"(
    
        // Vertex program
        uniform mat4 modelViewProjectionMatrix;
        uniform float spriteRadius;
        uniform mat4 cameraMatrix;
        in vec4 position;
        out vec2 spriteCoord; // sprite coordinate in [ -1 ,1]
        uniform int rainyClouds;
    
        void main () {
    
        vec4 pos = position ;
        switch ( gl_VertexID % 4) {
        case 0: spriteCoord = vec2 ( -1. , -1.) ; break ;
        case 1: spriteCoord = vec2 ( 1. , -1.) ; break ;
        case 2: spriteCoord = vec2 ( 1. , 1.) ; break ;
        case 3: spriteCoord = vec2 ( -1. , 1.) ; break ;
        }
    
        vec4 offset = vec4 ( spriteCoord * spriteRadius , 0. , 0.) ;
        offset = cameraMatrix * offset;
        pos += offset;
        gl_Position = modelViewProjectionMatrix * pos;
        }
    
    )", R"(
        // Fragment program
        in vec2 spriteCoord; // sprite coordinate in [ -1 ,1]
        out vec4 fragColor;
        uniform sampler2D noiseTex; // bring noise to shader
    
        void main () {
            float rsqr = dot(spriteCoord, spriteCoord);
            if(rsqr > 1.) discard;
    
            vec3 noiseColor = texture(noiseTex, (spriteCoord + vec2(1.0)) * 0.3).rgb; // creates a texture with noise
    
            vec3 col = vec3(0.90, 0.90, 0.90);
            float a = 1. - rsqr ; // inverted parabola
            float w = 0.5; // attenuation width
            float wsqr = w*w;
            a *= wsqr /( wsqr + rsqr );
            col *= a;
            fragColor = vec4 ( col * noiseColor, 1.); // applies noise to colour
        }
    )");
    
    /*--------------------------------------------------------------
     SET SPRITES
     --------------------------------------------------------------*/
         // Define position for each rectangle
         glm::vec3 position1(-2., 1.5, -2);
         glm::vec3 position2(-0.5, 1.8, 0);
         glm::vec3 position3(1., 1.5, -1);

         // Add rectangle to spriteMesh at the specified position
         addRect(spriteMesh, position1, 4., 2.);
         addRect(spriteMesh, position2, 4., 2.);
         addRect(spriteMesh, position3, 4., 2.);

    /*--------------------------------------------------------------
     LOAD  IMAGES AS TEXTURES
     --------------------------------------------------------------*/
    if (!sphereTex.load ("tex1.jpg")) std :: cout << " Error loading image file " << std :: endl;
    
    if (!torusTex.load ("tex2.jpg")) std :: cout << " Error loading image file " << std :: endl;
    
    if (!coneTex.load ("tex3.jpg")) std :: cout << " Error loading image file " << std :: endl;
    
    if (!cylinderTex.load ("tex4.jpg")) std :: cout << " Error loading image file " << std :: endl;
    
    /*--------------------------------------------------------------
     LIGHT AND MATERIAL SHADER
     --------------------------------------------------------------*/
    build(shader, R"(
        // Vertex shader code
        // Define uniforms for transformation matrices
        uniform mat4 modelViewProjectionMatrix;
        uniform mat4 projectionMatrix;
        uniform mat4 viewMatrix;
        uniform mat4 modelMatrix;
        
        // Define vertex attributes
        in vec4 position;       // Vertex position attribute
        in vec3 normal;         // Vertex normal attribute
        in vec2 texcoord;       // Texture coordinate attribute
        out vec2 vtexcoord;     // Interpolated texture coordinate passed to fragment shader
        out vec3 vposition;     // Interpolated vertex position passed to fragment shader
        out vec3 vnormal;       // Interpolated vertex normal passed to fragment shader
        in vec3 color;          // Vertex color attribute
        out vec3 vcolor;        // Interpolated vertex color passed to fragment shader
        
        void main() {
            // Pass vertex attributes to fragment shader
            vcolor = color;
            vtexcoord = texcoord;
            vnormal = normal;
            vposition = (modelMatrix * position).xyz;  // Transform vertex position to world space
            
            // Transform vertex position to clip space
            gl_Position = projectionMatrix * viewMatrix * vec4(vposition, 1.);
        }
    )", glslLighting() + R"(
        // Fragment shader code
        // Define uniforms and varying variables from vertex shader
        uniform vec3 eye;               // Camera position
        in vec3 vposition;              // Interpolated vertex position from vertex shader
        in vec3 vnormal;                // Interpolated vertex normal from vertex shader
        in vec3 vcolor;                 // Interpolated vertex color from vertex shader
        out vec4 fragColor;             // Final fragment color
        
        // Define texture and texturing variable
        uniform sampler2D tex;          // Texture (ID) passed in from the CPU
        in vec2 vtexcoord;              // Interpolated texture coordinate from vertex shader
        uniform float texturing;        // Texturing toggle variable
        
        void main() {
            // Compute lighting and apply texturing
            vec3 pos = vposition;           // Position of the fragment
            vec3 normal = normalize(vnormal);   // Normalized surface normal
            vec3 texColor = texture(tex, vtexcoord).rgb;   // Get texel color
            vec3 col = vcolor;              // Vertex color
            
            // Define material properties
            Material mtrl;
            mtrl.diffuse = texColor;        // Diffuse color from texture
            mtrl.specular = vec3(0.5);      // Specular color
            mtrl.shine = 100.;               // Shininess
            
            // Define lights
            Light sunLight;
            sunLight.pos = vec3(0., 3., 0);         // Light position
            sunLight.strength = 2.;               // Light strength
            sunLight.halfDist = 1.;               // Half distance for attenuation
            sunLight.ambient = 1.5;               // Ambient intensity
            sunLight.diffuse = vec3(1., 1., 1.);
            sunLight.specular = sunLight.diffuse;   // Light specular color
            
            // Define additional lights with similar properties
            Light umbrella = sunLight;
            umbrella.pos = vec3(1., -1.5, 0.);
            umbrella.diffuse = vec3(1., 0.6, 0.0);
            umbrella.specular = umbrella.diffuse;
            
            Light beam = sunLight;
            beam.pos = vec3(-1., -1.5, 0.5);
            beam.diffuse = vec3(0.9, 0.3, 0.3);
            beam.specular = beam.diffuse;
            
            // Compute light fall-off
            LightFall fall = computeLightFall(pos, normal, eye, sunLight, mtrl);
            addTo(fall, computeLightFall(pos, normal, eye, umbrella, mtrl));
            addTo(fall, computeLightFall(pos, normal, eye, beam, mtrl));
            
            // Compute final color and mix with vertex color based on texturing
            vec3 finalColor = lightColor(fall, mtrl);
            finalColor = mix(col, finalColor, texturing);   // Mix between vertex color and final color
            fragColor = vec4(finalColor, 1.0);  // Set final fragment color
        }
    )");
    
    /*--------------------------------------------------------------
     SHAPES MESH
     --------------------------------------------------------------*/
    
    /*------------------------------------------------------------------------*/
    /* -----------                 SPHERE                    -----------------*/
    /*------------------------------------------------------------------------*/
    sphereMesh = ofMesh :: sphere(0.1, 80);
    
    auto & tex1 = sphereTex.getTexture ();
    // Set wrapping modes for x and y directions
    tex1.setTextureWrap(GL_MIRRORED_REPEAT, GL_MIRRORED_REPEAT);
    // Scale the texture coordinates
    for( auto & tc1 : sphereMesh.getTexCoords ()){
        tc1 *= vec2 (8. ,4.) ;
    }
    // Set white color for each vertex of the sphere
    for( auto & p : sphereMesh.getVertices ())
        sphereMesh.addColor ( ofFloatColor (1 ,1 ,1));
    
    // Move the sphere to the left
    glm::vec3 spherePosition(0, 0, 0);
    for(auto &vertex : sphereMesh.getVertices()) {
        vertex += spherePosition;
    }
    /*------------------------------------------------------------------------*/
    /* -----------                   TORUS                   -----------------*/
    /*------------------------------------------------------------------------*/
    
    // Define parameters for the torus mesh
    int Nx = 64, Ny = Nx;
    torusMesh.setMode( OF_PRIMITIVE_TRIANGLES);
    
    // Generate vertices and texture coordinates for the torus mesh
    for( int j=0; j<Ny; ++j){
        for( int i=0; i<Nx; ++i){
            auto uv = vec2 (i,j)/( vec2 (Nx ,Ny) -1.f);
            auto pos = vec3 (uv , 0.);
            torusMesh.addVertex (pos);
            torusMesh.addTexCoord (uv);
            // Add triangle faces of rectangle if we are not on right or top edge
            if(j <(Ny -1) && i <(Nx -1) ){
                auto k = j*Nx + i;
                torusMesh.addIndex (k); // left - bottom triangle
                torusMesh.addIndex (k+1);
                torusMesh.addIndex (k+Nx);
                torusMesh.addIndex (k+Nx); // right - top triangle
                torusMesh.addIndex (k+1);
                torusMesh.addIndex (k+Nx +1);
            }
        }
    }
    // Generate positions and normals for the torus mesh
    for( auto & p : torusMesh.getVertices ()){
        auto t = 2 * 355./113. * p.x;
        float r = 0.06;
        float R = 0.15;
        float pi = 355. / 113.;
        
        auto N = vec3(
                      (std::sin(2 * pi * (1. - p.y))) * cos(t),
                      (std::sin(2 * pi * (1. - p.y))) * sin(t),
                      (std::cos(2 * pi * (1. - p.y)))
                      );
        
        torusMesh.addNormal(N);
        p = vec3(
                 (r * std::sin(2 * pi * (1. - p.y)) + R) * cos(t),
                 (r * std::sin(2 * pi * (1. - p.y)) + R) * sin(t),
                 (r * std::cos(2 * pi * (1. - p.y)))
                 );
        
        
        p = { p.x + 0.5 ,  p.y  , p.z };
    }
    
    
    // Get the texture of the torus
    auto& tex2 = torusTex.getTexture();
    
    // Set wrapping modes for the texture in x and y directions
    tex2.setTextureWrap(GL_REPEAT, GL_REPEAT);
    
    // Scale the texture coordinates
    for (auto& tc2 : torusMesh.getTexCoords()) {
        tc2 *= vec2(8., 4.);
    }
    
    // Move the torus to the right and up
    glm::vec3 torusPosition(0, 0, 0);
    for (auto& vertex : torusMesh.getVertices()) {
        vertex += torusPosition;
    }
    //        addNormalLines(normalsMesh, torusMesh);
    
    /*------------------------------------------------------------------------*/
    /* -----------            UMBRELLA (CONE)                -----------------*/
    /*------------------------------------------------------------------------*/
    
    int Nr = 32, Ntheta = 32;
    coneMesh.setMode(OF_PRIMITIVE_TRIANGLES);

    // Generate vertices and texture coordinates
    for (int j = 0; j < Nr; ++j) {
        for (int i = 0; i < Ntheta; ++i) {
            auto uv = vec2(i, j) / (vec2(Ntheta, Nr) - 1.f);
            auto pos = vec3(0.5 * uv.x * cos(2 * PI * uv.y), 0.5 * uv.x * sin(2 * PI * uv.y), 0.5 * uv.x);
            coneMesh.addVertex(pos);
            coneMesh.addTexCoord(uv);
            // Add triangle faces if we are not on the last ring or last theta
            if (j < (Nr - 1) && i < (Ntheta - 1)) {
                auto k = j * Ntheta + i;
                coneMesh.addIndex(k); // current vertex
                coneMesh.addIndex(k + 1); // next vertex on the same ring
                coneMesh.addIndex(k + Ntheta); // next vertex on the next ring
                coneMesh.addIndex(k + Ntheta); // next vertex on the next ring
                coneMesh.addIndex(k + 1); // next vertex on the same ring
                coneMesh.addIndex(k + Ntheta + 1); // next vertex on the next ring
            }
        }
    }

    // Generate positions and normals
    for (auto& p : coneMesh.getVertices()) {
        auto phi = atan2(p.y, p.x);
        auto theta = acos(p.z / 0.5);
        float r = 0.02;
        auto N = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
        coneMesh.addNormal(N);
    }

    // Get the texture of the umbrella
    auto& tex3 = coneTex.getTexture();

    // Set wrapping modes for the texture in x and y directions
    tex3.setTextureWrap(GL_REPEAT, GL_REPEAT);

    // Scale the texture coordinates
    for (auto& tc : coneMesh.getTexCoords()) {
        tc *= vec2(8., 4.);
    }
    
    for (auto& vertex :coneMesh.getVertices()) {
        vertex += conePosition;
    }

    /*------------------------------------------------------------------------*/
    /* -----------            UMBRELLA (STEAM)               -----------------*/
    /*------------------------------------------------------------------------*/
    
    int x1 = 32, y1 = 8;
    cylinderMesh.setMode(OF_PRIMITIVE_TRIANGLES);
    for (int j = 0; j < y1; ++j) {
        for (int i = 0; i < x1; ++i) {
            auto uv = vec2(i, j) / (vec2(x1, y1) - 1.f);
            auto pos = vec3(uv, 0.);
            cylinderMesh.addVertex(pos);
            cylinderMesh.addTexCoord(uv);
            // Add triangle faces of rectangle if we are not on right or top edge
            if (j < (y1 - 1) && i < (x1 - 1)) {
                auto k = j * x1 + i;
                cylinderMesh.addIndex(k); // left - bottom triangle
                cylinderMesh.addIndex(k + 1);
                cylinderMesh.addIndex(k + x1);
                cylinderMesh.addIndex(k + x1); // right - top triangle
                cylinderMesh.addIndex(k + 1);
                cylinderMesh.addIndex(k + x1 + 1);
            }
        }
    }

    // Generate positions and normals
    for (auto& p : cylinderMesh.getVertices()) {
        auto t = 2. * 355. / 113. * p.x;
        float r = 0.03; // radius
        float h = 0.5; // hight

        auto N = vec3(std::cos(t), 2. * p.y - 1, -std::sin(t));
        cylinderMesh.addNormal(N);
        p = vec3(
            r * std::cos(t),
            r * std::sin(t),
            h * (2. * p.y - 1.)
        );
        p = { p.x,p.z - 2.,-p.y };
    }
    
    // Get the texture of the umbrella
    auto& tex4 = cylinderTex.getTexture();

    // Set wrapping modes for the texture in x and y directions
    tex4.setTextureWrap(GL_REPEAT, GL_REPEAT);

    // Scale the texture coordinates
    for (auto& tc : cylinderMesh.getTexCoords()) {
        tc *= vec2(8., 4.);
    }
    
    for (auto& vertex : cylinderMesh.getVertices()) {
        vertex += cylinderPosition;
    }
    
    /*------------------------------------------------------------------------*/
    /* -----------                  TERRAIN                  -----------------*/
    /*------------------------------------------------------------------------*/
    int Nx1 = 32, Ny1 = Nx1;
    planeMesh.setMode(OF_PRIMITIVE_TRIANGLES);
    
    for (int j = 0; j < Ny1; ++j) {
        for (int i = 0; i < Nx1; ++i) {
            // Generate vertices and UV coordinates for a tessellated plane
            auto uv = vec2(i, j) / (vec2(Nx1, Ny1) - 1.f);
            auto pos = vec3(uv.x, uv.y, 0.); // Adjusting the x component
            planeMesh.addVertex(pos);
            planeMesh.addTexCoord(uv);

            // Add triangle faces of the rectangle
            if (j < (Ny1 - 1) && i < (Nx1 - 1)) {
                auto k = j * Nx1 + i;
                planeMesh.addIndex(k);
                planeMesh.addIndex(k + 1);
                planeMesh.addIndex(k + Nx1);
                planeMesh.addIndex(k + Nx1);
                planeMesh.addIndex(k + 1);
                planeMesh.addIndex(k + Nx1 + 1);
            }
        }
    }
    // Generate terrain by applying a periodic wave function to each vertex
    auto pwave = [](vec2 pos, vec2 freq) {
        auto pi = 355. / 113.;
        auto t = pi * dot(pos, freq);
        return vec3(
            std::cos(t),
            freq * pi * -std::sin(t)
        );
    };
    
    // generate mesh for terrain surface
    for (auto &p : planeMesh.getVertices()) {
        p = p * 3.f - 2.f;
        auto terrainInfo = calculateTerrain(vec2(p.x, p.y));
        p.z = terrainInfo.height * 0.4;// * 4 to decrease waves
        auto N = normalize(vec3(-terrainInfo.gradient.x, -terrainInfo.gradient.y, 1.));
        planeMesh.addNormal(N);

        // Color mapping based on terrain height
        vec3 col1(0.9, 0.7, 0.5); // lighter gold colour
        vec3 col2(0.6, 0.4, 0.1); // darker gold colour
        auto sigmoid = [](float x) { return x / (1. + std::abs(x)); };
        float c = sigmoid(p.z / 0.25) * 0.5 + 0.5;
        auto col = (col2 - col1) * c + col1;
        planeMesh.addColor(ofFloatColor(col.r, col.g, col.b));
        
    }
   
}
//--------------------------------------------------------------
void ofApp::update(){
    
    /*--------------------------------------------------------------
     SPRITES ANIMATION (CLOUDS)
     --------------------------------------------------------------*/
    
    // mve sprites in counterclockwise direction
    float speed = 2.;
    float time = ofGetElapsedTimef() * speed;
    float radius = 0.005;
    
    for (int i = 0; i < spriteMesh.getNumVertices(); ++i) {
        glm::vec3 pos = spriteMesh.getVertex(i);
        pos.x += cos(time + i) * radius;
        pos.y += sin(time + i) * radius;
        
        spriteMesh.setVertex(i, pos);
    }
    
    // Delta seconds of last frame render
    float dt = ofGetLastFrameTime();
    
    /*--------------------------------------------------------------
     OBJECTS ANIMATION
     // Update all the animation ramps if the flag indicating key press is set
     --------------------------------------------------------------*/

    if (startAnimation) {
        // Update positions of sphere, torus, and clouds by delta seconds
        for(auto& a : {&spherePosX, &spherePosY, &torusPosX, &torusPosY, &cloudsPos})
            a->update(dt);
        
        // the movement of x and y decrease over time
        // *gravity effect
        scale -= 0.008;
        
        if (wind) {
            // frequencies determinate how hight it goes and how fast moves to the right
            spherePosY.freq(0.4);
            spherePosX.freq(0.04);
            float para = spherePosY.para(); // PRamp ani: Map current phase to parabolic wave, in [0,1]
            float sphereX = spherePosX.phaseRad(); // PRamp ani: Get phase, in [0, 2pi)
            float sphereY = scale * std::sin(para); // parabolic mov to hight
            
            // Update sphere position
            spherePosition = glm::vec3(sphereX - 1., sphereY - 0.1, 0);
            
            torusPosX.freq(0.05);
            torusPosY.freq(0.07);
            float paraT = torusPosY.para(); // PRamp ani: Map current phase to parabolic wave, in [0,1]
            float torusY = scale * std::sin(paraT); // parabolic mov to hight
            float torusX = torusPosX.phaseRad(); // PRamp ani: Get phase, in [0, 2pi)
            
            // Update torus position
            torusPosition = glm::vec3(torusX -1.5, torusY -0.1, -0.3);
        }
        
        if (cyclone) {
            //  PRamp ani: Get unit Archimedean spiral with equal sampling per arc length
            auto sRotation = spherePosY.spiral<glm::vec2>(2.0);// 2 spin per cycle
            sphereRotation = ofRadToDeg(std::atan2(sRotation.y, sRotation.x)); // convert to degrees
            
            torusPosY.freq(0.4); // increase the spiral effect
            auto tRotation = torusPosY.spiral<glm::vec2>(1.);
            torusRotation = ofRadToDeg(std::atan2(tRotation.y, tRotation.x));
        }
        
        // stops animation
        if (scale < 0.0) {
            startAnimation = false;
        }
    }

}
//--------------------------------------------------------------
void ofApp::draw(){
    
    /*--------------------------------------------------------------
     DRAW MESHES, APPLY ANIMATION and TEXTURES
     --------------------------------------------------------------*/
    // sand floor
    cam.begin();
    shader.begin();
    shader.setUniform3f("eye", cam.getPosition());
    shader.setUniform1f("texturing", 0);
    ofPushMatrix();
    ofTranslate(-0.3, -0.2, 0);
    ofRotateDeg(180, 0, 1, 1);
    planeMesh.draw();
    ofPopMatrix();
    
    // top of the umbrella
    ofPushMatrix();
    shader.setUniform1f("texturing", 1);
    shader.setUniformTexture("tex3", coneTex, 0);
    ofTranslate(conePosition);
    ofRotateDeg(90, 0.9, -0.5, 0.2);
    coneMesh.draw();
    ofPopMatrix();
    
    // steam of the umbrella
    ofPushMatrix();
    shader.setUniform1f("texturing", 1);
    shader.setUniformTexture("tex4", cylinderTex, 0);
    ofTranslate(cylinderPosition);
    ofRotateDeg(45, 0.0, 0.2, -0.2);
    cylinderMesh.draw();
    ofPopMatrix();
    
    // beach ball
    shader.setUniform1f("texturing", 1);
    shader.setUniformTexture("tex1", sphereTex, 0);
    ofPushMatrix();
    ofTranslate(spherePosition);
    ofRotateDeg(sphereRotation, 1, 1, 0);
    sphereMesh.draw();
    ofPopMatrix();
      
    // buoy
    shader.setUniform1f("texturing", 1);
    shader.setUniformTexture("tex2", torusTex, 0); // Use the same texture for both sphere and torus
    ofPushMatrix();
    ofTranslate(torusPosition);
    ofRotateDeg(torusRotation, 1, 1, 1);
    torusMesh.draw();
    ofPopMatrix();
    
    shader.end();
    
    /*--------------------------------------------------------------
    DRAW SHADER SPRITES SEPARATED FROM LIGHT
     --------------------------------------------------------------*/
    glDepthMask(GL_FALSE);
    ofEnableBlendMode(OF_BLENDMODE_ADD);
    shaderSprites.begin();
    shaderSprites.setUniformMatrix4f("cameraMatrix", cam.getLocalTransformMatrix());
    shaderSprites.setUniform1f("spriteRadius", 0.1);
    shaderSprites.setUniformTexture("noiseTex", tex, 0);
    spriteMesh.draw();
    shaderSprites.end();
    ofDisableBlendMode();
    glDepthMask(GL_TRUE);
    
    cam.end();
    
    /*--------------------------------------------------------------
    SCREEN INSTRUCTIONS
     --------------------------------------------------------------*/
    ofSetColor(ofColor::orange);
    font.drawString("Breezy Beach Bliss", 100, 130);
    ofSetColor(ofColor::whiteSmoke);
    font.drawString("W for a bit of wind", 100, 160);
    font.drawString("C for cyclone mode", 100, 190);
    font.drawString("R to restart", 100, 220);
}

/*--------------------------------------------------------------
 TERRAIN
 // Function to calculate terrain information at a given position
 --------------------------------------------------------------*/
terrainNumbers ofApp::calculateTerrain(glm::vec2 pos) {
    // Function to generate a periodic wave
    auto pwave = [](vec2 pos, vec2 freq) {
        auto pi = 355. / 113.;
        auto t = pi * dot(pos, freq);
        return vec3(
            std::cos(t),
            freq * pi * -std::sin(t)
        );
    };

    // Map the position to the range [-1, 1] to center the surface
    pos = pos * 2.f - 1.f;

    // Calculate two periodic waves
    auto wave = 0.2 * pwave(pos, cos(vec2(2, 3)));
    wave += 0.1 * pwave(pos, sin(vec2(1, 4))); // add wave

    // Store the height and gradient of the terrain at the given position
    terrainNumbers terrain;
    terrain.height = wave.x;

    // Calculate gradient using partial derivatives
    terrain.gradient.x = -wave.y;
    terrain.gradient.y = -wave.z;

    return terrain;
}

/*--------------------------------------------------------------
 RESTART
 // Function to reset all animation parameters to their initial values
 --------------------------------------------------------------*/
void ofApp::resetAnimations() {
    startAnimation = false;
    wind = false;
    cyclone = false;
    scale = 2.0;
    spherePosition = glm::vec3(-1., -0.1, 0);
    torusPosition = glm::vec3(-1.5, -0.1, -0.3);
    sphereRotation = 0;
    torusRotation = 0;
    spherePosX.reset();
    spherePosY.reset();
    torusPosX.reset();
    torusPosY.reset();
    cloudsPos.reset();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    if (key == 'w') {
        // Set movement direction to right
        wind = true;
        startAnimation = true;
        
    } else if(key == 'c'){
        wind = true;
        cyclone = true;
        startAnimation = true;
        
    } else if(key == 'r') {
        // Reset all animations and parameters
        resetAnimations();
    }
}



