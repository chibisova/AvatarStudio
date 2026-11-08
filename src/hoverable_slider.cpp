#include "hoverable_slider.h"

HoverableSlider::HoverableSlider(Qt::Orientation orientation, QWidget* parent)
    : QSlider(orientation, parent)
{
    setMouseTracking(true);
}

void HoverableSlider::enterEvent(QEnterEvent* event)
{
    emit hovered();
    QSlider::enterEvent(event);
}

void HoverableSlider::leaveEvent(QEvent* event)
{
    emit unhovered();
    QSlider::leaveEvent(event);
}