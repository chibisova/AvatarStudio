#include "camera.h"
#include <QtGlobal>

Camera::Camera()
{
    distance = 250.0f;
    yaw = 0.0f;
    pitch = 0.0f;
    target = QVector3D(0.0f, 0.9f, 0.0f);
    defaultTarget = QVector3D(0.0f, 0.9f, 0.0f);
}

QMatrix4x4 Camera::viewMatrix() const
{
    QMatrix4x4 view;

    view.translate(0, 0, -distance);
    view.rotate(pitch, 1, 0, 0);
    view.rotate(yaw, 0, 1, 0);
    view.translate(-target); // we use negative value since we transform world into camera space

    return view;
}

QMatrix4x4 Camera::projectionMatrix(
    float aspectRatio,
    float fov,
    float nearPlane,
    float farPlane
    ) const
{
    QMatrix4x4 projection;

    projection.perspective(
        fov,
        aspectRatio,
        nearPlane,
        farPlane
        );

    return projection;
}

void Camera::setTarget(const QVector3D& newTarget)
{
    target = newTarget;
    defaultTarget = newTarget;
}

QVector3D Camera::targetPosition() const
{
    return target;
}

float Camera::distanceValue() const
{
    return distance;
}

void Camera::reset()
{
    distance = 250.0f;
    yaw = 0.0f;
    pitch = 0.0f;
    target = defaultTarget;
}

void Camera::orbit(float deltaYaw, float deltaPitch)
{
    yaw += deltaYaw;
    pitch += deltaPitch;

    pitch = qBound(-89.0f, pitch, 89.0f);
}

void Camera::zoom(float delta)
{
    distance -= delta;
    distance = qBound(0.5f, distance, 1000.0f);
}

void Camera::pan(float deltaX, float deltaY)
{
    target.setX(target.x() + deltaX);
    target.setY(target.y() + deltaY);
}