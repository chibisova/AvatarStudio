#include "objloader.h"

#include <QFile>
#include <QTextStream>
#include <QStringList>
#include <QRegularExpression>

bool OBJLoader::load(const QString& filePath)
{
    m_vertices.clear();
    m_normals.clear();
    m_triangleVertices.clear();
    m_triangleNormals.clear();

    bool firstVertex = true;

    QFile file(filePath);

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        return false;
    }

    QTextStream in(&file);

    while (!in.atEnd())
    {
        QString line = in.readLine();

        // Process verticies
        if (line.startsWith("v "))
        {
            QStringList parts = line.split(QRegularExpression("\\s+"));

            float x = parts[1].toFloat();
            float y = parts[2].toFloat();
            float z = parts[3].toFloat();

            QVector3D vertex(x, y, z);
            m_vertices.push_back(vertex);

            if (firstVertex)
            {
                m_minBounds = vertex;
                m_maxBounds = vertex;
                firstVertex = false;
            }
            else
            {
                m_minBounds.setX(qMin(m_minBounds.x(), x));
                m_minBounds.setY(qMin(m_minBounds.y(), y));
                m_minBounds.setZ(qMin(m_minBounds.z(), z));

                m_maxBounds.setX(qMax(m_maxBounds.x(), x));
                m_maxBounds.setY(qMax(m_maxBounds.y(), y));
                m_maxBounds.setZ(qMax(m_maxBounds.z(), z));
            }
        }

        // Process normals
        else if (line.startsWith("vn "))
        {
            QStringList parts = line.split(QRegularExpression("\\s+"));

            QVector3D normal(
                parts[1].toFloat(),
                parts[2].toFloat(),
                parts[3].toFloat()
                );

            m_normals.push_back(normal);
        }

        // Process faces
        else if (line.startsWith("f "))
        {
            QStringList parts = line.split(QRegularExpression("\\s+"));

            QStringList p1 = parts[1].split("/");
            QStringList p2 = parts[2].split("/");
            QStringList p3 = parts[3].split("/");

            int i1 = p1[0].toInt() - 1;
            int i2 = p2[0].toInt() - 1;
            int i3 = p3[0].toInt() - 1;

            // Handle faces without normals: f 3998 3896 5230
            if (p1.size() == 1 &&
                p2.size() == 1 &&
                p3.size() == 1)
            {
                int i1 = p1[0].toInt() - 1;
                int i2 = p2[0].toInt() - 1;
                int i3 = p3[0].toInt() - 1;

                addTriangle(i1, i2, i3);

                // Continue to the next face
                continue;
            }

            int n1 = p1[2].toInt() - 1;
            int n2 = p2[2].toInt() - 1;
            int n3 = p3[2].toInt() - 1;

            addTriangle(i1, i2, i3, n1, n2, n3);

            // Quad: convert into a second triangle
            if (parts.size() == 5)
            {
                QStringList p4 = parts[4].split("/");

                int i4 = p4[0].toInt() - 1;

                int n4 = p4[2].toInt() - 1;

                addTriangle(i1, i3, i4, n1, n3, n4);
            }

            //qDebug() << "Face indices:" << i1 << i2 << i3;
            //qDebug() << "Face vertex count:" << parts.size() - 1;
        }

    }

    return true;
}

const std::vector<QVector3D>& OBJLoader::vertices() const
{
    return m_vertices;
}

const std::vector<QVector3D>& OBJLoader::triangleVertices() const
{
    return m_triangleVertices;
}

void OBJLoader::addTriangle(
    int i1, int i2, int i3,
    int n1, int n2, int n3
)
{
    const QVector3D& a = m_vertices[i1];
    const QVector3D& b = m_vertices[i2];
    const QVector3D& c = m_vertices[i3];

    m_triangleVertices.push_back(a);
    m_triangleVertices.push_back(b);
    m_triangleVertices.push_back(c);

    // Store normals
    m_triangleNormals.push_back(m_normals[n1]);
    m_triangleNormals.push_back(m_normals[n2]);
    m_triangleNormals.push_back(m_normals[n3]);
}

void OBJLoader::addTriangle(int i1, int i2, int i3)
{
    const QVector3D& v1 = m_vertices[i1];
    const QVector3D& v2 = m_vertices[i2];
    const QVector3D& v3 = m_vertices[i3];

    QVector3D normal = QVector3D::crossProduct(
                           v2 - v1,
                           v3 - v1
                           ).normalized();


    m_triangleVertices.push_back(v1);
    m_triangleVertices.push_back(v2);
    m_triangleVertices.push_back(v3);

    m_triangleNormals.push_back(normal);
    m_triangleNormals.push_back(normal);
    m_triangleNormals.push_back(normal);
}

const std::vector<QVector3D>& OBJLoader::triangleNormals() const
{
    return m_triangleNormals;
}

QVector3D OBJLoader::minBounds() const
{
    return m_minBounds;
}

QVector3D OBJLoader::maxBounds() const
{
    return m_maxBounds;
}