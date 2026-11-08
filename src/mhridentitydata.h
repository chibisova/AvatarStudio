#pragma once

#include <QString>
#include <QVector3D>

#include <cstdint>
#include <vector>

class MHRIdentityData
{
public:
    bool load(const QString& path);

    int vertexCount() const;
    int parameterCount() const;
    int faceCount() const;

    const std::vector<QVector3D>& baseVertices() const;
    const std::vector<QVector3D>& identityBasis() const;
    const std::vector<std::int32_t>& faces() const;

private:
    int m_vertexCount = 0;
    int m_parameterCount = 0;
    int m_faceCount = 0;

    std::vector<QVector3D> m_baseVertices;
    std::vector<QVector3D> m_identityBasis;
    std::vector<std::int32_t> m_faces;
};