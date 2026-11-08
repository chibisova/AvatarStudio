#include "mhridentitydata.h"

#include <QFile>
#include <QDataStream>
#include <QDebug>

bool MHRIdentityData::load(const QString& path)
{
    QFile file(path);

    if (!file.open(QIODevice::ReadOnly))
    {
        qWarning() << "Failed to open MHR file:" << path;
        return false;
    }

    QDataStream stream(&file);

    stream.setByteOrder(QDataStream::LittleEndian);

    quint32 vertexCount;
    quint32 parameterCount;
    quint32 faceCount;

    stream >> vertexCount;
    stream >> parameterCount;
    stream >> faceCount;

    if (stream.status() != QDataStream::Ok)
    {
        qWarning() << "Failed to read MHR file header.";
        return false;
    }

    if (vertexCount == 0 ||
        parameterCount == 0 ||
        faceCount == 0)
    {
        qWarning() << "Invalid MHR file header.";
        return false;
    }

    m_vertexCount = static_cast<int>(vertexCount);
    m_parameterCount = static_cast<int>(parameterCount);
    m_faceCount = static_cast<int>(faceCount);

    qDebug() << "MHR identity file:";
    qDebug() << "Vertices:" << m_vertexCount;
    qDebug() << "Parameters:" << m_parameterCount;
    qDebug() << "Faces:" << m_faceCount;

    m_baseVertices.resize(m_vertexCount);

    for (int i = 0; i < m_vertexCount; ++i)
    {
        float xyz[3];

        qint64 bytesRead = file.read(
            reinterpret_cast<char*>(xyz),
            sizeof(xyz)
            );

        if (bytesRead != sizeof(xyz))
        {
            qWarning() << "Failed to read base vertex" << i;
            return false;
        }

        m_baseVertices[i] = QVector3D(
            xyz[0],
            xyz[1],
            xyz[2]
            );
    }

    if (stream.status() != QDataStream::Ok)
    {
        qWarning() << "Failed to read base vertices.";
        return false;
    }

    qDebug() << "Loaded base vertices:" << m_baseVertices.size();
    qDebug() << "First vertex:" << m_baseVertices[0];

    m_identityBasis.resize(
        m_parameterCount * m_vertexCount
        );

    for (int i = 0; i < m_parameterCount * m_vertexCount; ++i)
    {
        float xyz[3];

        qint64 bytesRead = file.read(
            reinterpret_cast<char*>(xyz),
            sizeof(xyz)
            );

        if (bytesRead != sizeof(xyz))
        {
            qWarning() << "Failed to read identity basis" << i;
            return false;
        }

        m_identityBasis[i] = QVector3D(
            xyz[0],
            xyz[1],
            xyz[2]
            );
    }

    qDebug() << "Loaded identity basis:"
             << m_identityBasis.size();

    m_faces.resize(m_faceCount * 3);

    for (int i = 0; i < m_faceCount * 3; ++i)
    {
        std::int32_t index;

        qint64 bytesRead = file.read(
            reinterpret_cast<char*>(&index),
            sizeof(index)
            );

        if (bytesRead != sizeof(index))
        {
            qWarning() << "Failed to read face index" << i;
            return false;
        }

        m_faces[i] = index;
    }

    qDebug() << "Loaded face indices:"
             << m_faces.size();

    qDebug() << "First triangle:"
             << m_faces[0]
             << m_faces[1]
             << m_faces[2];

    return true;
}

int MHRIdentityData::vertexCount() const
{
    return m_vertexCount;
}

int MHRIdentityData::parameterCount() const
{
    return m_parameterCount;
}

int MHRIdentityData::faceCount() const
{
    return m_faceCount;
}

const std::vector<QVector3D>& MHRIdentityData::baseVertices() const
{
    return m_baseVertices;
}

const std::vector<QVector3D>& MHRIdentityData::identityBasis() const
{
    return m_identityBasis;
}

const std::vector<std::int32_t>& MHRIdentityData::faces() const
{
    return m_faces;
}