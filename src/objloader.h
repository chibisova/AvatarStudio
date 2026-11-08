#ifndef OBJLOADER_H
#define OBJLOADER_H

#include <vector>
#include <QVector3D>

class OBJLoader
{
public:
    bool load(const QString& filePath);

    const std::vector<QVector3D>& vertices() const;
    const std::vector<QVector3D>& triangleVertices() const;
    const std::vector<QVector3D>& triangleNormals() const;

    QVector3D minBounds() const;
    QVector3D maxBounds() const;

private:
    std::vector<QVector3D> m_vertices;
    std::vector<QVector3D> m_normals;
    std::vector<QVector3D> m_triangleVertices;
    std::vector<QVector3D> m_triangleNormals;

    void addTriangle(
        int i1, int i2, int i3,
        int n1, int n2, int n3
    );

    void addTriangle(int i1, int i2, int i3);

    QVector3D m_minBounds;
    QVector3D m_maxBounds;

};

#endif