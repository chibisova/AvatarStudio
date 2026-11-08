#ifndef HOVERABLE_SLIDER_H
#define HOVERABLE_SLIDER_H

#include <QSlider>

class HoverableSlider : public QSlider
{
    Q_OBJECT

public:
    explicit HoverableSlider(Qt::Orientation orientation, QWidget* parent = nullptr);

signals:
    void hovered();
    void unhovered();

protected:
    void enterEvent(QEnterEvent* event) override;
    void leaveEvent(QEvent* event) override;
};

#endif // HOVERABLE_SLIDER_H