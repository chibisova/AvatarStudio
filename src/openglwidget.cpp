#include "openglwidget.h"

#include <QDebug>
#include <QVector4D>
#include <QMatrix4x4>
#include <QTimer>
#include <QDir>
#include <algorithm>

OpenGLWidget::OpenGLWidget(QWidget *parent) : QOpenGLWidget(parent)
{
    shaderProgram = new QOpenGLShaderProgram();
    setFocusPolicy(Qt::StrongFocus);
}

void OpenGLWidget::initializeGL()
{

    initializeOpenGLFunctions(); // Initializes the Qt-provided OpenGL function interface

    glClearColor(0.08f, 0.09f, 0.11f, 1.0f); // clear and draw current frame

    if (!shaderProgram->addShaderFromSourceCode(
            QOpenGLShader::Vertex,
            R"(
            attribute vec3 position;
            attribute vec3 normal;
            attribute vec3 vertexColor;

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            uniform mat3 normalMatrix;

            varying vec3 vertexNormal;
            varying vec3 color;

            void main()
            {
                vertexNormal = normalMatrix * normal;

                gl_Position = projection * view * model * vec4(position, 1.0);

                color = vertexColor;
            }
        )"))
    {
        qDebug() << shaderProgram->log();
    }


    if (!shaderProgram->addShaderFromSourceCode(
            QOpenGLShader::Fragment,
            R"(
            uniform vec4 triangleColor;
            uniform vec3 lightDirection;

            varying vec3 vertexNormal;
            varying vec3 color;


            void main()
            {
                float brightness = max(
                    dot(normalize(vertexNormal), normalize(lightDirection)),
                    0.0
                );

                float ambient = 0.2;

                vec3 finalColor = color * (ambient + brightness);

                gl_FragColor = vec4(finalColor, triangleColor.a);
            }
            )"
            ))
    {
        qDebug() << shaderProgram->log();
    }

    qDebug() << "Working directory:" << QDir::currentPath();

    // Load MHR parameters
    if (!m_mhrData.load("../../models/mhr_identity.bin"))
    {
        qWarning() << "Failed to load MHR identity data.";
        return;
    }

    m_neutralVertices = m_mhrData.baseVertices();
    m_currentVertices = m_neutralVertices;

    // Load OBJ
    if (objLoader.load("../../models/mhr_identity0.obj"))
    {
        qDebug() << "OBJ loaded!";
        qDebug() << "Vertex count:" << objLoader.vertices().size();
    }
    else
    {
        qDebug() << "Failed to load OBJ!";
    }

    negativeLoaders.resize(3);
    positiveLoaders.resize(3);

    // Use the MHR binary mesh as the initial geometry.
    m_neutralVertices = m_mhrData.baseVertices();

    m_currentVertices.clear();
    m_currentVertices.reserve(m_mhrData.faces().size());

    for (std::int32_t index : m_mhrData.faces()) {
        m_currentVertices.push_back(m_neutralVertices[index]);
    }

    qDebug() << "Neutral:" << m_neutralVertices.size();
    qDebug() << "Vertices:" << objLoader.triangleVertices().size();
    qDebug() << "Normals:" << objLoader.triangleNormals().size();

    // Load min and max parameter models

    for (int i = 0; i < 3; ++i)
    {
        QString negativePath =
            QString("../../models/mhr_identity%1_-2.obj").arg(i);

        QString positivePath =
            QString("../../models/mhr_identity%1_+2.obj").arg(i);

        bool negativeLoaded =
            negativeLoaders[i].load(negativePath);

        bool positiveLoaded =
            positiveLoaders[i].load(positivePath);

        qDebug() << "I" << i
                 << "negative:" << negativeLoaded
                 << "positive:" << positiveLoaded;

        qDebug() << "  negative vertices:"
                 << negativeLoaders[i].triangleVertices().size();

        qDebug() << "  positive vertices:"
                 << positiveLoaders[i].triangleVertices().size();
    }

    // Calculate the bounding box of loaded model to calibrate camera view
    qDebug() << "Min bounds:" << objLoader.minBounds();
    qDebug() << "Max bounds:" << objLoader.maxBounds();

    QVector3D minBounds = objLoader.minBounds();
    QVector3D maxBounds = objLoader.maxBounds();

    QVector3D meshCenter = (minBounds + maxBounds) * 0.5f;

    camera.setTarget(meshCenter);

    qDebug() << "Mesh center:" << meshCenter;

    const auto& meshVertices = objLoader.triangleVertices();

    shaderProgram->bindAttributeLocation("position", 0);
    shaderProgram->bindAttributeLocation("normal", 1);
    shaderProgram->bindAttributeLocation("vertexColor", 2);

    if (!shaderProgram->link())
    {
        qDebug() << shaderProgram->log();
    }

    // Upload verticies
    glGenBuffers(1, &vbo);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glBufferData(
        GL_ARRAY_BUFFER,
        meshVertices.size() * sizeof(QVector3D),
        meshVertices.data(),
        GL_STATIC_DRAW
    );

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, 0);
    glEnableVertexAttribArray(0);
    glEnable(GL_DEPTH_TEST);


    // Calculate smooth normals for the initial mesh
    m_currentNormals = calculateSmoothNormals(
        m_mhrData.baseVertices(),
        m_mhrData.faces()
        );

    // Upload normals
    glGenBuffers(1, &normalVbo);

    glBindBuffer(GL_ARRAY_BUFFER, normalVbo);

    glBufferData(
        GL_ARRAY_BUFFER,
        m_currentNormals.size() * sizeof(QVector3D),
        m_currentNormals.data(),
        GL_STATIC_DRAW
        );

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, nullptr);

    glEnableVertexAttribArray(1);

    // Create vertex color buffer
    // Create vertex color buffer with a default white color for every vertex.
    std::vector<QVector3D> initialColors(
        m_currentVertices.size(),
        QVector3D(1.0f, 1.0f, 1.0f)
        );

    glGenBuffers(1, &colorVbo);
    glBindBuffer(GL_ARRAY_BUFFER, colorVbo);

    glBufferData(
        GL_ARRAY_BUFFER,
        initialColors.size() * sizeof(QVector3D),
        initialColors.data(),
        GL_DYNAMIC_DRAW
        );

    glVertexAttribPointer(
        2, 3, GL_FLOAT, GL_FALSE, sizeof(QVector3D), nullptr
        );

    glEnableVertexAttribArray(2);

    // Animate object's rotation

    QTimer *timer = new QTimer(this);

    connect(timer, &QTimer::timeout, this, QOverload<>::of(&OpenGLWidget::update));

    timer->start(16);

    qDebug() << "OpenGL Version:" << reinterpret_cast<const char *>(glGetString(GL_VERSION));
    qDebug() << "OpenGL Vendor:" << reinterpret_cast<const char *>(glGetString(GL_VENDOR));
    qDebug() << "OpenGL Renderer:" << reinterpret_cast<const char *>(glGetString(GL_RENDERER));
}

void OpenGLWidget::resizeGL(int width, int height)
{
    if (width <= 0 || height <= 0)
        return;

    glViewport(0, 0, width, height);
}


void OpenGLWidget::paintGL()
{
    if (m_meshDirty)
    {
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        glBufferData(
            GL_ARRAY_BUFFER,
            m_currentVertices.size() * sizeof(QVector3D),
            m_currentVertices.data(),
            GL_DYNAMIC_DRAW
            );

        glBindBuffer(GL_ARRAY_BUFFER, normalVbo);

        glBufferData(
            GL_ARRAY_BUFFER,
            m_currentNormals.size() * sizeof(QVector3D),
            m_currentNormals.data(),
            GL_DYNAMIC_DRAW
            );

        m_meshDirty = false;
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clears the frame

    shaderProgram->bind();



    shaderProgram->setUniformValue(
        "lightDirection",
        QVector3D(1.0f, 1.0f, 1.0f)
    );

    shaderProgram->setUniformValue(
        "triangleColor",
        QVector4D(1.0f, 1.0f, 1.0f, 1.0f)
    );

    QMatrix4x4 model;
    model.rotate(rotationAngle, 0.0f, 1.0f, 0.0f);

    QMatrix4x4 view = camera.viewMatrix();

    float aspectRatio = float(width()) / float(height());

    QMatrix4x4 projection =
        camera.projectionMatrix(
        aspectRatio,
        45.0f,
        0.01f,
        1000.f
        );

    QMatrix3x3 normalMatrix = model.normalMatrix();

    shaderProgram->setUniformValue("normalMatrix", normalMatrix);

    shaderProgram->setUniformValue("model", model);
    shaderProgram->setUniformValue("view", view);
    shaderProgram->setUniformValue("projection", projection);

    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(m_currentVertices.size()));

    // rotationAngle += 1.0f;

    if (rotationAngle >= 360.0f)
        rotationAngle -= 360.0f;

}

void OpenGLWidget::updateMeshVertices(
    const std::vector<QVector3D>& vertices)
{
    makeCurrent();

    m_currentVertices = vertices;

    // These vertices are already expanded into triangles.
    // Calculate flat normals here only if this function is used.
    std::vector<QVector3D> normals(vertices.size());

    for (size_t i = 0; i + 2 < vertices.size(); i += 3) {
        QVector3D normal = QVector3D::crossProduct(
                               vertices[i + 1] - vertices[i],
                               vertices[i + 2] - vertices[i]
                               ).normalized();

        normals[i] = normal;
        normals[i + 1] = normal;
        normals[i + 2] = normal;
    }

    m_currentNormals = normals;
    m_meshDirty = true;

    update();
    doneCurrent();
}

// Camera controls

void OpenGLWidget::wheelEvent(QWheelEvent *event)
{

    float delta = event->angleDelta().y() / 120.0f * 0.2f;
    camera.zoom(delta);

    update();
}

void OpenGLWidget::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton &&
        event->modifiers() & Qt::ShiftModifier)
    {
        isPanning = true;
        lastMousePosition = event->pos();
        return;
    }

    if (event->button() == Qt::LeftButton)
    {
        isDragging = true;
        lastMousePosition = event->pos();
    }

    if (event->button() == Qt::MiddleButton)
    {

        isPanning = true;
        lastMousePosition = event->pos();
    }

    setFocus();
}


void OpenGLWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (isPanning)
    {
        QPoint delta = event->pos() - lastMousePosition;

        camera.pan(-delta.x() * 0.01f,
                   delta.y() * 0.01f);

        lastMousePosition = event->pos();

        update();
        return;
    }

    if (!isDragging)
        return;

    QPoint delta = event->pos() - lastMousePosition;

    camera.orbit(delta.x() * 0.5f,
                 delta.y() * 0.5f);

    lastMousePosition = event->pos();

    update();
}

void OpenGLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton)
    {
        isDragging = false;
        isPanning = false;
    }

    if (event->button() == Qt::MiddleButton)
    {
        isPanning = false;
    }
}

void OpenGLWidget::keyPressEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_R)
    {
        camera.reset();
        update();
    }
}

// Interpolate mesh for Identity i
void OpenGLWidget::setIdentityParameter(int index, float value)
{
    if (index < 0 || index >= identityParameters.size())
        return;

    identityParameters[index].value = value;

    const int vertexCount = m_mhrData.vertexCount();

    std::vector<QVector3D> result = m_mhrData.baseVertices();

    for (int i = 0; i < vertexCount; ++i)
    {
        for (int parameterIndex = 0;
             parameterIndex < identityParameters.size();
             ++parameterIndex)
        {
            float parameterValue =
                identityParameters[parameterIndex].value;

            const QVector3D& basis =
                m_mhrData.identityBasis()[
                    parameterIndex * vertexCount + i
            ];

            result[i] += basis * parameterValue;
        }
    }

    std::vector<QVector3D> renderVertices;
    renderVertices.reserve(m_mhrData.faces().size());

    for (std::int32_t index : m_mhrData.faces())
    {
        renderVertices.push_back(result[index]);
    }

    m_currentNormals = calculateSmoothNormals(
        result,
        m_mhrData.faces()
        );

    m_currentVertices = renderVertices;
    m_meshDirty = true;
    update();
}

std::vector<QVector3D> OpenGLWidget::calculateSmoothNormals(
    const std::vector<QVector3D>& vertices,
    const std::vector<std::int32_t>& faces)
{
    // One accumulated normal per original vertex.
    std::vector<QVector3D> vertexNormals(vertices.size());

    // Accumulate each triangle's normal at its three vertices.
    for (size_t i = 0; i + 2 < faces.size(); i += 3)
    {
        int i0 = faces[i];
        int i1 = faces[i + 1];
        int i2 = faces[i + 2];

        const QVector3D& v0 = vertices[i0];
        const QVector3D& v1 = vertices[i1];
        const QVector3D& v2 = vertices[i2];

        QVector3D faceNormal =
            QVector3D::crossProduct(v1 - v0, v2 - v0);

        vertexNormals[i0] += faceNormal;
        vertexNormals[i1] += faceNormal;
        vertexNormals[i2] += faceNormal;
    }

    // Normalize the accumulated normals.
    for (auto& normal : vertexNormals)
    {
        if (normal.lengthSquared() > 1e-12f)
            normal.normalize();
    }

    // Expand normals in exactly the same order as the triangle list.
    std::vector<QVector3D> triangleNormals;
    triangleNormals.reserve(faces.size());

    for (std::int32_t index : faces)
        triangleNormals.push_back(vertexNormals[index]);

    return triangleNormals;
}

int OpenGLWidget::identityParameterCount() const
{
    return static_cast<int>(identityParameters.size());
}

const OpenGLWidget::IdentityParameter&
OpenGLWidget::identityParameter(int index) const
{
    return identityParameters[index];
}

// we get index of slider as an input and need to calculate influence of it on every vertex
std::vector<float> OpenGLWidget::calculateInfluence(int index)
{
    const int vertexCount = m_mhrData.vertexCount();
    std::vector<float> influence(vertexCount);

    if (index >= 0 && index < identityParameters.size()) {

        for (int i = 0; i < vertexCount; ++i){

            const QVector3D& basis =
                m_mhrData.identityBasis()[index * vertexCount + i];

            influence[i] = basis.length();
        }

        if (influence.empty()){
            return influence;
        } else {

            // Normalize against the maximum displacement magnitude.
            // This preserves differences in overall influence between parameters.
            auto maxIt = std::max_element(influence.begin(), influence.end());

            if (maxIt == influence.end() || *maxIt <= 1e-8f) {
                std::fill(influence.begin(), influence.end(), 0.0f);
            } else {
                const float maxInfluence = *maxIt;

                for (auto& value : influence) {
                    value = std::clamp(value / maxInfluence, 0.0f, 1.0f);
                }
            }
        }

    } else {
        influence = {};
    }

    return influence;
}

QVector3D OpenGLWidget::influenceToColor(float influence)
{

    QVector3D blue = QVector3D(0.0f, 0.0f, 1.0f);
    QVector3D cyan = QVector3D(0.0f, 1.0f, 1.0f);
    QVector3D green = QVector3D(0.0f, 1.0f, 0.0f);
    QVector3D yellow = QVector3D(1.0f, 1.0f, 0.0f);
    QVector3D red = QVector3D(1.0f, 0.0f, 0.0f);

    float clamp_inlfuence = std::clamp(influence, 0.0f, 1.0f);

    float scaled = clamp_inlfuence * 4.0f;

    QVector3D A = blue;
    QVector3D B = cyan;

    int segment = 0;
    float t = 0.0f;

    if (scaled < 1){
        segment = 0;
        A = blue;
        B = cyan;
    } else if (scaled < 2) {
        segment = 1;
        A = cyan;
        B = green;
    } else if (scaled < 3) {
        segment = 2;
        A = green;
        B = yellow;
    } else {
        segment = 3;
        A = yellow;
        B = red;
    }

    t = scaled - segment;

    return A + t*(B-A);
}

std::vector<QVector3D> OpenGLWidget::calculateHeatmapColors(int index) {

    std::vector<float> influence = calculateInfluence(index);

    if (influence.empty()) {
        return {};
    }

    std::vector<QVector3D> colors;
    colors.reserve(m_mhrData.faces().size());

    for (std::int32_t i : m_mhrData.faces())
    {
        if (i < 0 || static_cast<std::size_t>(i) >= influence.size()) {
            return {};
        }

        colors.push_back(influenceToColor(influence[i]));
    }

    return colors;
}

void OpenGLWidget::setHeatmapParameter(int index)
{
    m_currentColors = calculateHeatmapColors(index);

    if (m_currentColors.empty())
        return;

    makeCurrent();

    glBindBuffer(GL_ARRAY_BUFFER, colorVbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        m_currentColors.size() * sizeof(QVector3D),
        m_currentColors.data(),
        GL_DYNAMIC_DRAW
        );

    doneCurrent();
    update();
}

void OpenGLWidget::clearHeatmap()
{
    m_currentColors.assign(
        m_currentVertices.size(),
        QVector3D(1.0f, 1.0f, 1.0f)
        );

    makeCurrent();

    glBindBuffer(GL_ARRAY_BUFFER, colorVbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        m_currentColors.size() * sizeof(QVector3D),
        m_currentColors.data(),
        GL_DYNAMIC_DRAW
        );

    doneCurrent();
    update();
}
