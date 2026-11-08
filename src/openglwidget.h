#ifndef OPENGLWIDGET_H
#define OPENGLWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QWheelEvent>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QString>
#include <vector>

#include "OBJLoader.h"
#include "camera.h"
#include "mhridentitydata.h"

class OpenGLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
public:
    explicit OpenGLWidget(QWidget *parent = nullptr);
    void setIdentityParameter(int index, float value);
    int identityParameterCount() const;

    struct IdentityParameter
    {
        QString name;
        float value;
        float min;
        float max;
    };

    const IdentityParameter& identityParameter(int index) const;

    std::vector<float> calculateInfluence(int index);
    QVector3D influenceToColor(float influence);
    std::vector<QVector3D> calculateHeatmapColors(int index);

    void setHeatmapParameter(int index);
    void clearHeatmap();

protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;

    void wheelEvent(QWheelEvent *event) override;

    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

    void keyPressEvent(QKeyEvent *event) override;

    void updateMeshVertices(const std::vector<QVector3D>& vertices);

private:
    QOpenGLShaderProgram *shaderProgram;
    GLuint vbo;
    GLuint normalVbo;
    GLuint colorVbo;

    OBJLoader objLoader;

    std::vector<OBJLoader> negativeLoaders;
    std::vector<OBJLoader> positiveLoaders;

    Camera camera;

    float rotationAngle = 0.0f;
    QPoint lastMousePosition;
    bool isDragging = false;

    bool isPanning = false;

    std::vector<QVector3D> m_neutralVertices;
    std::vector<QVector3D> m_currentVertices;
    std::vector<QVector3D> m_currentNormals;
    std::vector<QVector3D> m_currentColors;

    std::vector<QVector3D> calculateSmoothNormals(
        const std::vector<QVector3D>& vertices,
        const std::vector<std::int32_t>& faces
        );

    std::vector<IdentityParameter> identityParameters = {
        {"I_0",  0.0f, -2.0f, 2.0f},
        {"I_1",  0.0f, -2.0f, 2.0f},
        {"I_2",  0.0f, -2.0f, 2.0f},
        {"I_3",  0.0f, -2.0f, 2.0f},
        {"I_4",  0.0f, -2.0f, 2.0f},
        {"I_5",  0.0f, -2.0f, 2.0f},
        {"I_6",  0.0f, -2.0f, 2.0f},
        {"I_7",  0.0f, -2.0f, 2.0f},
        {"I_8",  0.0f, -2.0f, 2.0f},
        {"I_9",  0.0f, -2.0f, 2.0f},
        {"I_10", 0.0f, -2.0f, 2.0f},
        {"I_11", 0.0f, -2.0f, 2.0f},
        {"I_12", 0.0f, -2.0f, 2.0f},
        {"I_13", 0.0f, -2.0f, 2.0f},
        {"I_14", 0.0f, -2.0f, 2.0f},
        {"I_15", 0.0f, -2.0f, 2.0f},
        {"I_16", 0.0f, -2.0f, 2.0f},
        {"I_17", 0.0f, -2.0f, 2.0f},
        {"I_18", 0.0f, -2.0f, 2.0f},
        {"I_19", 0.0f, -2.0f, 2.0f},
        {"I_20", 0.0f, -2.0f, 2.0f},
        {"I_21", 0.0f, -2.0f, 2.0f},
        {"I_22", 0.0f, -2.0f, 2.0f},
        {"I_23", 0.0f, -2.0f, 2.0f},
        {"I_24", 0.0f, -2.0f, 2.0f},
        {"I_25", 0.0f, -2.0f, 2.0f},
        {"I_26", 0.0f, -2.0f, 2.0f},
        {"I_27", 0.0f, -2.0f, 2.0f},
        {"I_28", 0.0f, -2.0f, 2.0f},
        {"I_29", 0.0f, -2.0f, 2.0f},
        {"I_30", 0.0f, -2.0f, 2.0f},
        {"I_31", 0.0f, -2.0f, 2.0f},
        {"I_32", 0.0f, -2.0f, 2.0f},
        {"I_33", 0.0f, -2.0f, 2.0f},
        {"I_34", 0.0f, -2.0f, 2.0f},
        {"I_35", 0.0f, -2.0f, 2.0f},
        {"I_36", 0.0f, -2.0f, 2.0f},
        {"I_37", 0.0f, -2.0f, 2.0f},
        {"I_38", 0.0f, -2.0f, 2.0f},
        {"I_39", 0.0f, -2.0f, 2.0f},
        {"I_40", 0.0f, -2.0f, 2.0f},
        {"I_41", 0.0f, -2.0f, 2.0f},
        {"I_42", 0.0f, -2.0f, 2.0f},
        {"I_43", 0.0f, -2.0f, 2.0f},
        {"I_44", 0.0f, -2.0f, 2.0f}
    };

    bool m_meshDirty = false;

    MHRIdentityData m_mhrData;
};

#endif // OPENGLWIDGET_H