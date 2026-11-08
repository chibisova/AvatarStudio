#pragma once

#include <QMatrix4x4>
#include <QVector3D>

class Camera
{
public:
    Camera();

    QMatrix4x4 viewMatrix() const;

    void reset();
    void orbit(float deltaYaw, float deltaPitch);
    void zoom(float delta);
    void pan(float deltaX, float deltaY);
    void setTarget(const QVector3D& newTarget);

    QVector3D targetPosition() const;
    float distanceValue() const;

    QMatrix4x4 projectionMatrix(    // Transforms 3D coordinates into the screen's
                                    // coordinate space, enabling perspective

        float aspectRatio,          // Vieport width ÷ height
        float fov = 45.0f,          // Field of view in degrees
        float nearPlane = 0.1f,     // Closest visible distance
        float farPlane = 100.0f     // Farthest visible distance
    ) const;

private:
    float distance;
    float yaw;
    float pitch;

    QVector3D defaultTarget;
    QVector3D target;
};